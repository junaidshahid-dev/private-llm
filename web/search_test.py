"""search_test.py — web_search intelligence: parse, dedup, rank, multi-query, gating.

    python web/search_test.py

The DDG HTML parse is pinned by a fixture; dedup/rank/multi-query are pure logic; the live search is
validated in the end-of-phase runs. No network here.
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

from web.search import parse_ddg_html, dedup, rank, web_search                # noqa: E402

fails = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))
    if not ok:
        fails.append(name)


# a realistic slice of DuckDuckGo HTML (uddg-wrapped url + title + snippet)
FIXTURE = """
<div class="result results_links">
  <a rel="nofollow" class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fnmap.org%2Fbook%2F">
    Nmap Reference Guide</a>
  <a class="result__snippet" href="x">The official Nmap book and reference for scanning.</a>
</div>
<div class="result results_links">
  <a rel="nofollow" class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fpage">
    Some Example Page</a>
  <a class="result__snippet" href="y">An unrelated example page about cooking.</a>
</div>
"""


def main() -> int:
    print("=" * 70)
    print("WEB_SEARCH TEST — parse, dedup, rank, multi-query, gating")
    print("=" * 70)

    print("\n1. parse_ddg_html (fixture)")
    res = parse_ddg_html(FIXTURE)
    check("finds both results", len(res) == 2, str(len(res)))
    check("recovers the real URL from uddg", res[0]["url"] == "https://nmap.org/book/",
          res[0]["url"])
    check("extracts the title", res[0]["title"] == "Nmap Reference Guide")
    check("extracts the snippet", "reference" in res[0]["snippet"].lower())
    check("tags source=web", res[0]["source"] == "web")

    print("\n2. dedup — by normalized URL")
    dupes = [{"url": "https://a.com/x/", "title": "1"}, {"url": "http://a.com/x", "title": "2"},
             {"url": "https://b.com", "title": "3"}]
    check("collapses trailing-slash / scheme dupes", len(dedup(dupes)) == 2)

    print("\n3. rank — query-term overlap")
    ranked = rank(res, "nmap scanning reference")
    check("the on-topic nmap result ranks first", "nmap" in ranked[0]["url"], ranked[0]["url"])

    print("\n4. web_search — permission gate + multi-query merge")
    off = web_search({"web": {"enabled": False}}, "q")
    check("disabled group denied", not off["ok"])
    off2 = web_search({"web": {"enabled": True, "search": False}}, "q")
    check("search not permitted denied", not off2["ok"])

    cfg = {"web": {"enabled": True, "search": True}}
    calls = []

    def backend(q):
        calls.append(q)
        return FIXTURE
    out = web_search(cfg, "nmap reference", queries=["nmap reference", "nmap book"], _search=backend)
    check("multi-query hits the backend per query", len(calls) == 2, str(calls))
    check("results merged, deduped, ranked", out["ok"] and out["result"]["count"] == 2)
    check("nmap result ranked first", "nmap" in out["result"]["results"][0]["url"])
    check("labelled untrusted, source=web", out["source"] == "web" and "UNTRUSTED" in out["note"])

    def boom(q):
        raise RuntimeError("backend down")
    check("a failing backend is a soft error, not a crash",
          not web_search(cfg, "q", _search=boom)["ok"])

    from web.search import SearchBackendBlocked

    def blocked(q):
        raise SearchBackendBlocked("DDG returned HTTP 202 with an empty body — blocked.")
    bout = web_search(cfg, "q", _search=blocked)
    check("a BLOCKED backend is flagged, not reported as 0 results",
          not bout["ok"] and bout.get("backend_blocked") is True)

    # wiring
    from mcp_layer import controller
    check("web_search in the tool surface", "web_search" in controller._all_tool_names())

    print("\n" + "=" * 70)
    if fails:
        print(f"FAILED {len(fails)}: {', '.join(fails)}")
        return 1
    print("ALL WEB_SEARCH TESTS PASS — parse/dedup/rank/multi-query/gating; live search validated later.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
