"""web_test.py — the web layer: SSRF-safe validation, web_fetch, and MCP wiring.

    python web/web_test.py

Mock resolver + mock fetch, so the SSRF boundary (public-only, redirect re-validation) is tested
without touching the network.
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

from web.safety import is_public_ip, validate_public_url                      # noqa: E402
from web.fetch import web_fetch                                              # noqa: E402

fails = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))
    if not ok:
        fails.append(name)


def resolver(mapping):
    def r(host):
        if host in mapping:
            return mapping[host]
        raise OSError("nxdomain")
    return r


def mock_fetch(responses):
    def f(url, timeout, max_bytes):
        return responses.get(url, {"status": 404, "headers": {}, "body": b"nf"})
    return f


CFG = {"web": {"enabled": True, "fetch": True, "private_networks": False}}


def main() -> int:
    print("=" * 74)
    print("WEB LAYER TEST — SSRF default-deny, redirect re-validation, untrusted labelling")
    print("=" * 74)

    print("\n1. is_public_ip")
    for ip in ("93.184.216.34", "8.8.8.8", "1.1.1.1"):
        check(f"{ip} public", is_public_ip(ip))
    for ip in ("127.0.0.1", "10.0.0.5", "192.168.1.9", "169.254.169.254", "172.16.0.1"):
        check(f"{ip} NOT public (blocked)", not is_public_ip(ip))

    print("\n2. validate_public_url — resolves then checks the IP")
    ok, *_ = validate_public_url("http://example.com/x", resolver({"example.com": "93.184.216.34"}))
    check("public host allowed", ok)
    ok2, _, _, why = validate_public_url("http://intranet/x", resolver({"intranet": "10.0.0.5"}))
    check("host resolving to PRIVATE ip blocked (SSRF)", not ok2 and "non-public" in why)
    ok3, _, _, why3 = validate_public_url("http://meta/x", resolver({"meta": "169.254.169.254"}))
    check("host resolving to cloud-metadata blocked", not ok3)
    check("scheme file:// blocked", not validate_public_url("file:///etc/passwd")[0])
    check("unresolvable host blocked",
          not validate_public_url("http://nope.invalid/", resolver({}))[0])

    print("\n3. web_fetch — permission gate")
    check("disabled group denied", not web_fetch({"web": {"enabled": False}}, "http://x")["ok"])
    check("fetch not permitted denied",
          not web_fetch({"web": {"enabled": True, "fetch": False}}, "http://x")["ok"])

    print("\n4. web_fetch — fetch a public page, label it untrusted")
    r = web_fetch(CFG, "http://example.com/", _resolver=resolver({"example.com": "93.184.216.34"}),
                  _fetch=mock_fetch({"http://example.com/":
                                     {"status": 200, "headers": {"Content-Type": "text/html"},
                                      "body": b"<html>hello world</html>"}}))
    check("public fetch ok", r["ok"] and "hello world" in r["result"]["body"])
    check("tagged source=web", r.get("source") == "web")
    check("body labelled UNTRUSTED", "UNTRUSTED" in r["note"])

    print("\n5. web_fetch — redirect re-validation (rebinding/open-redirect)")
    res = web_fetch(CFG, "http://good.com/go",
                    _resolver=resolver({"good.com": "93.184.216.34", "evil.com": "93.184.216.34"}),
                    _fetch=mock_fetch({
                        "http://good.com/go": {"status": 302, "headers":
                                               {"Location": "http://good.com/ok"}, "body": b""},
                        "http://good.com/ok": {"status": 200, "headers":
                                               {"Content-Type": "text/plain"}, "body": b"landed"}}))
    check("redirect to a public host followed", res["ok"] and "landed" in res["result"]["body"])
    bad = web_fetch(CFG, "http://good.com/jump",
                    _resolver=resolver({"good.com": "93.184.216.34", "inner": "127.0.0.1"}),
                    _fetch=mock_fetch({"http://good.com/jump": {"status": 302, "headers":
                                       {"Location": "http://inner/secret"}, "body": b""}}))
    check("redirect to a host resolving PRIVATE is blocked", not bad["ok"] and "blocked" in
          bad["error"])

    print("\n6. MCP wiring — controller sees web_fetch")
    from mcp_layer import controller
    check("web_fetch in the tool surface", "web_fetch" in controller._all_tool_names())
    check("proposal kind is 'web'", controller.parse_proposals(
        '{"tool":"web_fetch","arguments":{"url":"http://x"}}')[0]["kind"] == "web")
    check("schema advertises web_fetch",
          "web_fetch" in controller._available_tools_text())

    print("\n" + "=" * 74)
    if fails:
        print(f"FAILED {len(fails)}: {', '.join(fails)}")
        return 1
    print("ALL WEB TESTS PASS — public-only fetch, redirects re-validated, content untrusted, wired.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
