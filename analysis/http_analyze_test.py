"""http_analyze_test.py — HTTP response security analysis + attack-surface extraction.

    python analysis/http_analyze_test.py
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

from analysis.http_analyze import analyze_response, attack_surface           # noqa: E402

fails = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))
    if not ok:
        fails.append(name)


def issues(findings):
    return {f["issue"] for f in findings}


def main() -> int:
    print("=" * 70)
    print("HTTP ANALYZE — security headers, cookies, disclosure, CORS, attack surface")
    print("=" * 70)

    print("\n1. missing security headers on an https response")
    f = analyze_response(200, {"Server": "Apache/2.4.25", "Set-Cookie": "sid=abc"},
                         "<html>ok</html>", "https://lab.local/")
    i = issues(f)
    check("flags missing CSP", "missing_content-security-policy" in i)
    check("flags missing HSTS on https", "missing_hsts" in i)
    check("flags server version disclosure", "server_version_disclosure" in i)
    check("flags cookie without HttpOnly", "cookie_no_httponly" in i)
    check("flags cookie without Secure over https", "cookie_no_secure" in i)

    print("\n2. permissive CORS with credentials is HIGH")
    c = analyze_response(200, {"Access-Control-Allow-Origin": "*",
                               "Access-Control-Allow-Credentials": "true"}, "", "https://lab.local/")
    hit = next((x for x in c if x["issue"] == "cors_wildcard"), None)
    check("CORS wildcard + credentials flagged HIGH", hit and hit["severity"] == "HIGH", str(hit))

    print("\n3. body info-disclosure")
    b = analyze_response(500, {}, "You have an error in your SQL syntax near ...", "https://x/")
    check("flags SQL error disclosure", any(x["issue"] == "info_disclosure" for x in b))

    print("\n4. a well-secured response is (nearly) clean")
    secure = analyze_response(200, {
        "Content-Security-Policy": "default-src 'self'", "X-Frame-Options": "DENY",
        "X-Content-Type-Options": "nosniff", "Referrer-Policy": "no-referrer",
        "Strict-Transport-Security": "max-age=63072000",
        "Set-Cookie": "sid=abc; HttpOnly; Secure; SameSite=Strict"}, "<html>ok</html>", "https://x/")
    check("secured response has no header/cookie findings", secure == [], str(secure))

    print("\n5. attack surface — forms + params")
    s = attack_surface('<form action="/login" method="post"><input name="user"><input name="pass">'
                       '</form>', "https://x/search?q=1&page=2")
    check("extracts the form action + method", s["forms"] and s["forms"][0]["action"] == "/login"
          and s["forms"][0]["method"] == "POST")
    check("extracts the input names", set(s["forms"][0]["inputs"]) == {"user", "pass"})
    check("extracts url params", set(s["url_params"]) == {"q", "page"})

    print("\n" + "=" * 70)
    if fails:
        print(f"FAILED {len(fails)}: {', '.join(fails)}")
        return 1
    print("ALL HTTP-ANALYZE TESTS PASS — headers/cookies/CORS/disclosure flagged, surface extracted.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
