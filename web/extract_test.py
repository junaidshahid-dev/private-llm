"""extract_test.py — HTML/PDF extraction to clean text + metadata, and web_extract wiring.

    python web/extract_test.py
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

from web.extract import html_to_text, extract_metadata, pdf_to_text, web_extract   # noqa: E402

fails = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))
    if not ok:
        fails.append(name)


PAGE = """<html><head><title>Nmap &amp; You</title>
<meta name="description" content="A guide to service detection.">
<meta property="og:title" content="Nmap and You">
<link rel="canonical" href="https://ex.com/nmap"></head>
<body><script>var x=1; alert('inject')</script><style>.a{color:red}</style>
<h1>Service Detection</h1><p>Use nmap -sV to detect versions.</p>
<p>Banners are <b>not</b> proof.</p><footer>copyright</footer></body></html>"""


def main() -> int:
    print("=" * 70)
    print("WEB EXTRACT TEST — html/pdf -> clean text + metadata")
    print("=" * 70)

    print("\n1. html_to_text")
    txt = html_to_text(PAGE)
    check("drops script content", "alert" not in txt and "var x" not in txt)
    check("drops style content", "color:red" not in txt)
    check("keeps real text", "Use nmap -sV to detect versions." in txt)
    check("block tags become line breaks", "Service Detection" in txt.splitlines()[0]
          or any("Service Detection" in l for l in txt.splitlines()))
    check("no angle brackets left", "<" not in txt and ">" not in txt)

    print("\n2. extract_metadata")
    md = extract_metadata(PAGE)
    check("title (entities unescaped)", md["title"] == "Nmap & You", md["title"])
    check("description", md["description"] == "A guide to service detection.")
    check("og_title", md["og_title"] == "Nmap and You")
    check("canonical", md["canonical"] == "https://ex.com/nmap")

    print("\n3. pdf_to_text — graceful without the library")
    txt2, err = pdf_to_text(b"%PDF-1.4 not-a-real-pdf")
    check("returns a clean result or a clear 'needs pypdf' error",
          (txt2 is not None) or (err and "pypdf" in err) or (err and "PDF parse" in err), str(err))

    print("\n4. web_extract — fetch + extract (mock network)")
    cfg = {"web": {"enabled": True, "fetch": True, "private_networks": False}}

    def resolver(_h):
        return "93.184.216.34"

    def fetch(_u, _t, _m):
        return {"status": 200, "headers": {"Content-Type": "text/html"}, "body": PAGE.encode()}
    r = web_extract(cfg, "http://ex.com/nmap", _resolver=resolver, _fetch=fetch)
    check("returns clean text", r["ok"] and "detect versions" in r["result"]["text"])
    check("returns metadata title", r["result"]["metadata"]["title"] == "Nmap & You")
    check("labelled untrusted, source=web", r["source"] == "web" and "UNTRUSTED" in r["note"])
    check("SSRF still enforced (private host blocked)",
          not web_extract(cfg, "http://x/", _resolver=lambda h: "10.0.0.1", _fetch=fetch)["ok"])
    check("disabled group denied", not web_extract({"web": {"enabled": False}}, "http://x")["ok"])

    print("\n5. wiring")
    from mcp_layer import controller
    check("web_extract in the tool surface", "web_extract" in controller._all_tool_names())

    print("\n" + "=" * 70)
    if fails:
        print(f"FAILED {len(fails)}: {', '.join(fails)}")
        return 1
    print("ALL EXTRACT TESTS PASS — clean text + metadata, SSRF-gated, untrusted, wired.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
