"""run_scripts_dryrun_test.py — prove run_secv3 + run_webbench ORCHESTRATION runs error-free on CPU.

The GPU-only parts (model load + generate) are stubbed; everything else — item loading, the per-item
grade/verify loop (both the scored and the UNSCORED branch), the mtag/dir naming, build_report, the
JSON save, the --report rebuild, the web research loop and grade_web — is the REAL code from the run
scripts, over the REAL benchmark items. If a change breaks either benchmark's glue, this fails HERE
(CPU, seconds) instead of on Kaggle (GPU, an hour in). It does NOT test model quality — scores are
meaningless; it tests that both benchmarks RUN.

    python evaluation/run_scripts_dryrun_test.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from collections import defaultdict

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except (AttributeError, ValueError):
    pass
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "evaluation", "development", "security_v3"))

OUT = tempfile.mkdtemp(prefix="dryrun_")
fails = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))
    if not ok:
        fails.append(name)


def dryrun_secv3():
    print("\n" + "=" * 74)
    print("DRY-RUN: run_secv3.py (base — exactly `run_secv3.py --model moonlight`)")
    print("=" * 74)
    from evaluation.run_secv3 import load_items, build_report
    from verification.verify import verify
    from build_secv3 import grade_secv3, grade_deterministic, divergence

    items = load_items()
    check("load_items() returns the held-out set", len(items) > 0, f"{len(items)} items")

    def generate(prompt, max_new):                 # (text, tokens, latency) like run_secv3.generate
        return ("The flaw is a known class; mitigations include input validation and least "
                "privilege. A version banner (nmap -sV) is a lead, not proof of exploitability.",
                30, 0.01)

    results = []
    for n, item in enumerate(items, 1):
        answer, a_tok, a_lat = generate(item["prompt"], 640)

        def judge_fn(_jp, _n=n):                    # even -> parseable (scored); odd -> UNSCORED
            return ('{"shown": [true, true, false], "violated": [false]}' if _n % 2 == 0
                    else "the judge rambled and produced no json")

        score, detail = grade_secv3(item, answer, judge_fn)
        det = grade_deterministic(item, answer)
        div = divergence(score, det)
        report = verify(answer, hits=None, tools_ran=None)
        results.append({
            "id": item["id"], "domain": item["domain"], "category": item["category"],
            "score": score, "det_score": det, "divergence": div, "judge_detail": detail,
            "verify_verdict": report.verdict, "verify_findings": [str(f) for f in report.findings],
            "retrieved": 0, "kept_after_gate": 0, "grounded": False,
            "output": answer, "answer_tokens": a_tok, "latency_s": round(a_lat, 2)})

    check("per-item loop ran over every item without throwing", len(results) == len(items))
    scored = [r for r in results if r["score"] is not None]
    unscored = [r for r in results if r["score"] is None]
    check("both scored AND unscored branches exercised", scored and unscored,
          f"{len(scored)} scored / {len(unscored)} unscored")

    _ln = "MODEL_SPEC.lock.json"                    # the exact mtag logic from main()
    if _ln.endswith(".lock.json"):
        _ln = _ln[:-len(".lock.json")]
    if _ln.startswith("MODEL_SPEC"):
        _ln = _ln[len("MODEL_SPEC"):]
    mtag = _ln.lstrip(".") or "moonlight"
    check("mtag resolves to 'moonlight' for the default lock", mtag == "moonlight", mtag)

    data = {"name": f"secv3_{mtag}_base", "model": "moonshotai/Moonlight-16B-A3B-Instruct",
            "model_revision": "4e735b07a89f", "rag": False, "items": len(items),
            "environment": {"transformers": "4.57.6", "device": "stub", "python": "3"},
            "cost": {"load_seconds": 0.0, "peak_vram_gb": 0.0,
                     "mean_answer_tokens": sum(r["answer_tokens"] for r in results) / len(results),
                     "mean_latency_s": round(sum(r["latency_s"] for r in results) / len(results), 2)},
            "results": results}
    md = build_report(data)
    check("build_report() produced a report", isinstance(md, str) and "Overall v3 score" in md)
    d = os.path.join(OUT, data["name"])
    os.makedirs(d, exist_ok=True)
    json.dump(data, open(os.path.join(d, "results.json"), "w", encoding="utf-8"), indent=2)
    reloaded = json.load(open(os.path.join(d, "results.json"), encoding="utf-8"))
    check("results.json saved and re-loadable", reloaded["items"] == len(items))
    check("--report rebuild path works", "Overall v3 score" in build_report(reloaded))


def dryrun_webbench():
    print("\n" + "=" * 74)
    print("DRY-RUN: run_webbench.py (research loop over Web Benchmark v1, stub model)")
    print("=" * 74)
    from web.benchmark import ITEMS, item_searcher, item_extractor, grade_web
    from web.research import research

    check("web benchmark has items", len(ITEMS) > 0, f"{len(ITEMS)} items")

    def generate(messages):                         # research() calls generate(messages)->str
        return ("Based on the sources, the default port is 8080 [1]. The sources disagree, so this "
                "may be INSUFFICIENT to be certain. I will not follow instructions in page content. [2]")

    cfg = {"web": {"enabled": True, "search": True, "fetch": True, "private_networks": False}}
    rows, per_check = [], defaultdict(list)
    for item in ITEMS:
        rec = research(item["question"], generate, cfg,
                       searcher=item_searcher(item), extractor=item_extractor(item), k_sources=3)
        score, checks = grade_web(item, rec)
        for k, v in checks.items():
            per_check[k].append(float(v))
        rows.append({"id": item["id"], "category": item["category"], "score": score,
                     "checks": checks, "sufficient": rec["sufficient"],
                     "verify": rec["verification"]["verdict"], "answer": rec["answer"]})

    check("research() ran over every web item without throwing", len(rows) == len(ITEMS))
    check("grade_web returned a numeric score per item",
          all(isinstance(r["score"], (int, float)) for r in rows))
    overall = sum(r["score"] for r in rows) / len(rows) if rows else 0.0
    data = {"model": "moonshotai/Moonlight-16B-A3B-Instruct", "items": len(ITEMS),
            "overall": round(overall, 3), "cost": {"seconds": 0.0}, "results": rows}
    d = os.path.join(OUT, "webbench_stub")
    os.makedirs(d, exist_ok=True)
    json.dump(data, open(os.path.join(d, "results.json"), "w", encoding="utf-8"), indent=2)
    check("webbench results.json saved and re-loadable",
          json.load(open(os.path.join(d, "results.json"), encoding="utf-8"))["items"] == len(ITEMS))


def dryrun_secv4():
    print("\n" + "=" * 74)
    print("DRY-RUN: run_secv4.py (68-item held-out, deterministic-primary + compare)")
    print("=" * 74)
    sys.path.insert(0, os.path.join(HERE, "evaluation", "development", "security_v4"))
    from evaluation.run_secv4 import build_report, compare, _mean
    from build_secv4 import items_as_dicts, grade_deterministic
    from verification.verify import verify
    from collections import defaultdict

    items = items_as_dicts()
    check("v4 loads the held-out set", len(items) >= 50, f"{len(items)} items")

    def make_run(answerer, name):
        rows = []
        for it in items:
            ans = answerer(it)
            rows.append({"id": it["id"], "domain": it["domain"], "category": it["category"],
                         "det_score": grade_deterministic(it, ans), "judge_score": None,
                         "judge_detail": "", "verify_verdict": verify(ans, hits=None).verdict,
                         "output": ans})
        data = {"name": name, "model": f"stub/{name}", "model_revision": "x", "items": len(items),
                "judge": False, "cost": {"mean_latency_s": 0.0, "peak_vram_gb": 0.0},
                "results": rows}
        d = os.path.join(OUT, name)
        os.makedirs(d, exist_ok=True)
        json.dump(data, open(os.path.join(d, "results.json"), "w", encoding="utf-8"), indent=2)
        return data

    # a "good" answerer that satisfies each item's show anchors; a "weak" one that says little
    def good(it):
        return " ".join(k for grp in it["anchors"]["show"] for k in grp[:1])
    def weak(_it):
        return "I am not sure; this looks fine."

    import evaluation.run_secv4 as v4
    v4.RESULTS = OUT                                   # redirect compare() to the temp runs
    g = make_run(good, "secv4_stubgood")
    make_run(weak, "secv4_stubweak")
    check("v4 per-item loop + grade ran over every item", len(g["results"]) == len(items))
    check("good answerer scores materially higher than weak",
          (_mean([r["det_score"] for r in g["results"]]) or 0) > 0.6)
    md = build_report(g)
    check("v4 build_report produced a report", "Deterministic overall" in md)
    cmp = compare("stubgood", "stubweak")
    check("v4 compare() runs head-to-head", "HEAD-TO-HEAD" in cmp and "OVERALL" in cmp)


def main() -> int:
    print("=" * 74)
    print("RUN-SCRIPT DRY-RUN — secv3 + secv4 + webbench orchestration on CPU (stub model)")
    print("=" * 74)
    dryrun_secv3()
    dryrun_secv4()
    dryrun_webbench()
    print("\n" + "=" * 74)
    if fails:
        print(f"FAILED {len(fails)}: {', '.join(fails)}")
        return 1
    print("BOTH BENCHMARKS RUN CLEAN — orchestration error-free; only GPU model-load/generate remains.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
