"""research_test.py — research mode orchestration: plan, gather, synthesize, insufficiency, verify.

    python web/research_test.py

Mocks the model, searcher, and extractor, so the whole plan->search->read->synthesize->verify
pipeline is tested without a network or GPU.
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

from web.research import plan_queries, gather_evidence, research               # noqa: E402

fails = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))
    if not ok:
        fails.append(name)


def searcher(results_by_query):
    def s(config, q):
        return {"ok": True, "result": {"results": results_by_query.get(q, [])}}
    return s


def extractor(text_by_url):
    def e(config, url):
        if url in text_by_url:
            return {"ok": True, "result": {"final_url": url, "text": text_by_url[url]}}
        return {"ok": False, "error": "fetch failed"}
    return e


def main() -> int:
    print("=" * 70)
    print("RESEARCH MODE TEST — plan/search/read/synthesize/verify, insufficiency honesty")
    print("=" * 70)

    print("\n1. plan_queries")
    qs = plan_queries("what is nmap -sV", lambda m: "nmap sV flag\nnmap version detection\n")
    check("parses one query per line", qs == ["nmap sV flag", "nmap version detection"], str(qs))
    check("falls back to the question if the model returns nothing",
          plan_queries("Q?", lambda m: "") == ["Q?"])
    check("strips numbering/bullets",
          plan_queries("q", lambda m: "1. alpha\n- beta") == ["alpha", "beta"])

    print("\n2. gather_evidence — search then read top-k, deduped")
    cfg = {"web": {"enabled": True, "search": True, "fetch": True}}
    s = searcher({"qone": [{"url": "http://a.com", "title": "A"}, {"url": "http://b.com", "title": "B"}],
                  "qtwo": [{"url": "http://a.com", "title": "A"}, {"url": "http://c.com", "title": "C"}]})
    ex = extractor({"http://a.com": "alpha text", "http://b.com": "beta text",
                    "http://c.com": "gamma text"})
    ev = gather_evidence(["qone", "qtwo"], cfg, 2, s, ex)
    check("stops at k sources", len(ev) == 2)
    check("deduped across queries (a.com once)",
          [e["url"] for e in ev].count("http://a.com") == 1)
    check("carries the extracted text", ev[0]["text"] == "alpha text")

    print("\n3. research — full pipeline with citations")
    gen_calls = []

    def gen(messages):
        gen_calls.append(messages)
        # first call = plan (must return queries the mock searcher knows); later = synthesis
        if "planning web research" in messages[0]["content"]:
            return "qone\nqtwo"
        return "Nmap -sV detects service versions [1]. Banners are not proof [2]."
    rec = research("what does nmap -sV do", gen, cfg, searcher=s, extractor=ex, k_sources=2)
    check("planned then searched then synthesized", len(gen_calls) == 2)
    check("has sources with urls", rec["evidence_count"] == 2 and rec["sources"][0]["url"])
    check("answer carries citations", "[1]" in rec["answer"])
    check("marked sufficient (evidence found)", rec["sufficient"] is True)
    check("verification ran", rec["verification"]["verdict"] in ("PASS", "WARNING", "BLOCK"))

    print("\n4. insufficiency — no evidence => say so, do NOT guess")
    empty = research("obscure question", gen, cfg,
                     searcher=searcher({}), extractor=extractor({}), k_sources=3)
    check("no sources gathered", empty["evidence_count"] == 0)
    check("marked insufficient", empty["sufficient"] is False)
    check("answer admits insufficiency, does not bluff",
          "insufficient" in empty["answer"].lower() and "not guess" in empty["answer"].lower())

    print("\n" + "=" * 70)
    if fails:
        print(f"FAILED {len(fails)}: {', '.join(fails)}")
        return 1
    print("ALL RESEARCH TESTS PASS — plans, gathers cited evidence, and admits when it lacks it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
