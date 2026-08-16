"""try_web_fetch.py — live-test web_fetch against a real PUBLIC URL.

    python scripts/try_web_fetch.py https://example.com/

Enables the web group for this one run and fetches read-only through the real web_fetch (SSRF-safe:
a private/localhost/metadata URL will be refused). For validation; production use goes through
configs/tools.yaml + operator approval.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from web.fetch import web_fetch                                              # noqa: E402

url = sys.argv[1] if len(sys.argv) > 1 else "https://example.com/"
cfg = {"web": {"enabled": True, "fetch": True, "private_networks": False}}
r = web_fetch(cfg, url)
if not r.get("ok"):
    print("web_fetch FAILED:", r.get("error"))
    sys.exit(1)
res = r["result"]
print(f"status {res['status']}   {res['content_type']}   final_url={res['final_url']}")
print(f"redirects: {[h['status'] for h in res['redirects']]}   truncated={res['truncated']}")
print("--- body (first 500 chars — UNTRUSTED web DATA) ---")
print(res["body"][:500])
print(f"\n[source={r.get('source')}] [{r['note']}]")
