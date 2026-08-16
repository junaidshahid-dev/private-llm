"""try_http_get.py — live-test the http_get tool against an authorized URL (e.g. your lab).

    python scripts/try_http_get.py http://127.0.0.1:8080/login.php

Builds a minimal config that authorizes ONLY the host in the URL you pass (an explicit operator
act — you are naming the target), then fetches it read-only through the real MCP http_get. This is
for quick lab validation; production use goes through configs/tools.yaml + the operator-approval
loop. It fetches nothing you did not name on the command line.
"""
from __future__ import annotations

import os
import sys
from urllib.parse import urlparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mcp_layer import security as sec                                        # noqa: E402

url = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8080/login.php"
host = urlparse(url).hostname
if not host:
    sys.exit(f"no host in URL: {url!r}")

cfg = {"security_tools": {"enabled": True, "require_confirmation": False,
                          "authorized_targets": [{"id": "cli", "match": host,
                                                  "note": "try_http_get: you named this target"}]}}
r = sec.http_get(cfg, url, confirmed=True)
if not r.get("ok"):
    print("http_get FAILED:", r.get("error"))
    sys.exit(1)
res = r["result"]
print(f"status {res['status']}   {res['content_type']}   final_url={res['final_url']}")
print(f"redirects: {[h['status'] for h in res['redirects']]}   truncated={res['truncated']}")
print("--- body (first 500 chars — UNTRUSTED DATA) ---")
print(res["body"][:500])
print(f"\n[{r['note']}]")
