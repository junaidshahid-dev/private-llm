"""test_security.py — the security-tool sandbox must scope to the lab and gate every call.

    python mcp_layer/test_security.py

The tests that matter are the REJECTIONS: a scanner that can reach a public IP or a host outside
your lab is a liability, not a feature. So this asserts, hard, that out-of-scope targets are
refused before anything runs, that a disabled group runs nothing, and that confirmation is
required — and only then that an in-scope target passes the gate. No nmap install needed: we test
the gate, not the scan.
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from mcp_layer import security as sec       # noqa: E402

fails = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))
    if not ok:
        fails.append(name)


def main() -> int:
    print("=" * 74)
    print("SECURITY TOOL LAYER — target validation + gating")
    print("=" * 74)
    allowed = ["127.0.0.1", "localhost", "192.168.1.0/24"]

    print("\n1. target validation — REJECTIONS (the point)")
    for bad, label in [("8.8.8.8", "public DNS"), ("1.1.1.1", "public IP"),
                       ("192.168.2.5", "different subnet"), ("10.0.0.5", "other RFC1918 range"),
                       ("example.com", "arbitrary hostname"), ("0.0.0.0", "wildcard"),
                       ("169.254.1.1", "link-local")]:
        ok, why = sec.target_allowed(bad, allowed)
        check(f"reject {bad} ({label})", not ok, why[:46])

    print("\n2. target validation — ALLOWED")
    for good, label in [("127.0.0.1", "loopback"), ("localhost", "listed hostname"),
                        ("192.168.1.1", "in-range"), ("192.168.1.254", "in-range edge")]:
        ok, why = sec.target_allowed(good, allowed)
        check(f"allow {good} ({label})", ok, why[:40])

    print("\n3. dispatch gating")
    cfg_off = {"security_tools": {"enabled": False, "allowed_targets": allowed,
                                  "require_confirmation": True}}
    r = sec.dispatch({"tool": "nmap_scan", "arguments": {"target": "127.0.0.1"}}, cfg_off)
    check("disabled group runs nothing", not r["ok"] and "disabled" in r["error"])

    cfg_on = {"security_tools": {"enabled": True, "allowed_targets": allowed,
                                 "require_confirmation": True},
              "filesystem_read": {"enabled": True, "allowed_paths": [HERE]}}
    # out-of-scope target: refused even though the group is enabled
    r = sec.dispatch({"tool": "nmap_scan", "arguments": {"target": "8.8.8.8"}}, cfg_on)
    check("enabled group still rejects out-of-scope target", not r["ok"]
          and "not allowed" in r["error"])
    # in-scope target without confirmation: proposes, does not run
    r = sec.dispatch({"tool": "nmap_scan", "arguments": {"target": "127.0.0.1"}}, cfg_on)
    check("in-scope target requires confirmation first", not r["ok"]
          and r.get("needs_confirmation") is True, "proposes, waits for approval")
    # in-scope + confirmed: passes the gate (then runs nmap, or reports it is not installed)
    r = sec.dispatch({"tool": "nmap_scan", "arguments": {"target": "127.0.0.1"}}, cfg_on,
                     confirmed=True)
    passed_gate = r.get("ok") or "not installed" in (r.get("error", "") or "")
    check("confirmed in-scope target passes the gate", passed_gate,
          r.get("error", "ran")[:40])

    print("\n4. audit log records attempts")
    before = os.path.getsize(sec.AUDIT_LOG) if os.path.exists(sec.AUDIT_LOG) else 0
    sec.dispatch({"tool": "nmap_scan", "arguments": {"target": "8.8.8.8"}}, cfg_on)
    after = os.path.getsize(sec.AUDIT_LOG) if os.path.exists(sec.AUDIT_LOG) else 0
    check("denied attempt was logged", after > before)

    print("\n" + "=" * 74)
    print(f"FAILED: {fails}" if fails else "ALL SECURITY-LAYER TESTS PASSED — lab scope holds.")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
