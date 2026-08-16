"""try_web_search.py — live-test web_search against DuckDuckGo (no API key).

    python scripts/try_web_search.py nmap version detection

Enables web.search for this one run and prints the ranked candidates. If the parse returns nothing,
DuckDuckGo's HTML likely drifted — the parser (web/search.py) is the thing to update; the
intelligence (dedup/rank/multi-query) is unaffected.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from web.search import web_search                                            # noqa: E402

query = " ".join(sys.argv[1:]) or "nmap version detection"
cfg = {"web": {"enabled": True, "search": True}}
r = web_search(cfg, query)
if not r.get("ok"):
    print("web_search FAILED:", r.get("error"))
    sys.exit(1)
res = r["result"]
print(f"query: {res['query']!r}   {res['count']} results (UNTRUSTED candidates)\n")
for i, hit in enumerate(res["results"], 1):
    print(f"{i}. {hit['title']}\n   {hit['url']}\n   {hit['snippet'][:140]}\n")
print(f"[{r['note'][:90]}...]")
