"""test_security.py — authorization is by the operator's explicit list, not by IP class.

    python mcp_layer/test_security.py

The design principle under test: target CLASS does not decide authorization. A public IP the
operator listed is allowed; a private IP they did not list is rejected; an expired authorization
stops working. And the model can never authorize a target itself — only the config does, and
deny-by-default means an empty list runs nothing.
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta, timezone

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from mcp_layer import security as sec       # noqa: E402

fails = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))
    if not ok:
        fails.append(name)


def main() -> int:
    print("=" * 76)
    print("SECURITY LAYER — operator authorization, not IP class")
    print("=" * 76)

    future = (datetime.now(timezone.utc) + timedelta(days=1)).isoformat()
    past = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
    authed = [
        {"id": "my-vps", "match": "203.0.113.10", "note": "my public VPS"},        # PUBLIC IP
        {"id": "home-lab", "match": "192.168.1.0/24", "note": "my lab"},
        {"id": "client", "match": "scanme.example.com", "note": "RoE signed", "expires": future},
        {"id": "old-job", "match": "198.51.100.5", "note": "expired", "expires": past},
    ]

    print("\n1. the principle: class does not decide — the LIST does")
    ok, why = sec.target_authorized("203.0.113.10", authed)
    check("PUBLIC IP that IS authorized -> allowed", ok, why[:48])
    ok, why = sec.target_authorized("8.8.8.8", authed)
    check("public IP NOT authorized -> rejected", not ok)
    ok, why = sec.target_authorized("192.168.5.5", authed)
    check("PRIVATE IP NOT authorized -> rejected (private != approved)", not ok)
    ok, why = sec.target_authorized("192.168.1.50", authed)
    check("private IP that IS authorized -> allowed", ok, why[:40])
    ok, why = sec.target_authorized("scanme.example.com", authed)
    check("authorized hostname -> allowed", ok)
    ok, why = sec.target_authorized("https://scanme.example.com:8443/x", authed)
    check("URL whose host is authorized -> allowed", ok, "host extracted from URL")
    ok, why = sec.target_authorized("evil.example.org", authed)
    check("unlisted hostname -> rejected", not ok)

    print("\n2. temporary authorization expires")
    ok, _ = sec.target_authorized("198.51.100.5", authed)
    check("expired authorization -> rejected", not ok)

    print("\n3. deny by default")
    ok, why = sec.target_authorized("203.0.113.10", [])
    check("empty authorized list -> nothing allowed", not ok, why[:44])

    print("\n4. dispatch gating (enabled + authorized + confirmed)")
    cfg = {"security_tools": {"enabled": True, "require_confirmation": True,
                              "authorized_targets": authed},
           "filesystem_read": {"enabled": True, "allowed_paths": [HERE]}}
    r = sec.dispatch({"tool": "nmap_scan", "arguments": {"target": "8.8.8.8"}}, cfg)
    check("unauthorized target refused even when group enabled", not r["ok"]
          and "not authorized" in r["error"])
    r = sec.dispatch({"tool": "nmap_scan", "arguments": {"target": "203.0.113.10"}}, cfg)
    check("authorized public IP still needs confirmation", r.get("needs_confirmation") is True)
    r = sec.dispatch({"tool": "nmap_scan", "arguments": {"target": "203.0.113.10"}}, cfg,
                     confirmed=True)
    check("authorized + confirmed passes the gate", r.get("ok") or "not installed" in
          (r.get("error", "") or ""), r.get("error", "ran")[:36])

    print("\n5. offline analysis tools work without touching a target")
    r = sec.dispatch({"tool": "url_info", "arguments":
                      {"url": "https://sub.example.com:8443/a/b?q=1"}}, cfg)
    check("url_info parses host/port offline",
          r["ok"] and r["result"]["host"] == "sub.example.com" and r["result"]["port"] == 8443)

    print("\n6. the model cannot self-authorize")
    check("no tool exists to add an authorization",
          all(t["name"] not in ("authorize", "add_target", "approve_target") for t in sec.schema()))

    print("\n" + "=" * 76)
    print(f"FAILED: {fails}" if fails else
          "ALL PASSED — operator authorization holds; class does not decide.")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
