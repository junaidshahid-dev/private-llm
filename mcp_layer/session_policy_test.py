"""session_policy_test.py — one operator authorization, then autonomous; hard gates hold.

    python mcp_layer/session_policy_test.py

Also enforces that EVERY tool declares the rich schema (no tool left out) — the policy relies on it.
"""
from __future__ import annotations

import os
import sys
import tempfile
import time

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except (AttributeError, ValueError):
    pass
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
os.environ["KILL_SWITCH_FILE"] = os.path.join(tempfile.mkdtemp(), ".KILL_SWITCH")

from mcp_layer.session_policy import (AuthorizedSession, PROFILES, authorize, approver_for,   # noqa: E402
                                      authorize_target, start_session, _tool_index)
from mcp_layer import killswitch                                            # noqa: E402
from mcp_layer.session import run_session                                   # noqa: E402

fails = []
_SE_CLASSES = {"none", "network:read", "network:write", "local:write"}

CFG = {
    "filesystem_read": {"enabled": True, "allowed_paths": [HERE]},
    "git_inspect": {"enabled": True, "allowed_repos": [HERE]},
    "security_tools": {"enabled": True, "require_confirmation": True,
                       "authorized_targets": [{"match": "192.168.56.0/24"}, {"match": "lab.local"}]},
    "web": {"enabled": True, "fetch": True, "search": True},
}


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))
    if not ok:
        fails.append(name)


def scripted(*replies):
    q = list(replies)
    return lambda messages: q.pop(0) if q else "final answer."


def recording_executor():
    calls = []

    def ex(proposal, config, operator_ack=False):
        calls.append({"tool": proposal.get("tool"), "operator_ack": operator_ack})
        return {"ok": True, "tool": proposal.get("tool"), "result": {"ran": True}}
    return ex, calls


def main() -> int:
    print("=" * 74)
    print("SESSION POLICY — one authorization then autonomous; kill/scope/profile gates hold")
    print("=" * 74)
    killswitch.clear(operator_ack=True)

    print("\n0. EVERY tool declares the rich schema (no tool left out)")
    idx = _tool_index()
    missing = []
    for name, e in idx.items():
        if not (isinstance(e.get("read_only"), bool) and e.get("side_effects") in _SE_CLASSES
                and "required_binary" in e and e.get("verification_method")
                and isinstance(e.get("capabilities"), list) and e["capabilities"]):
            missing.append(name)
    check(f"all {len(idx)} tools declare read_only/side_effects/required_binary/verification_method/"
          "capabilities", not missing, str(missing))

    print("\n1. only the OPERATOR can start a session (model text cannot)")
    check("start_session without operator_ack is refused",
          start_session("x", ["lab.local"], operator_ack=False, config=CFG)["ok"] is False)
    check("start_session(operator_ack='true' string) is refused",
          start_session("x", ["lab.local"], operator_ack="true", config=CFG)["ok"] is False)
    r = start_session("assess the lab web app", ["lab.local"], "recon", operator_ack=True, config=CFG)
    check("operator start (ack=True) succeeds", r["ok"] and r["session"].objective.startswith("assess"))
    sess = r["session"]

    print("\n2. start guards (the operator is the authority for their own scope)")
    check("the operator can authorize ANY target they declare (operator is the authority)",
          start_session("x", ["203.0.113.200"], operator_ack=True, config=CFG)["ok"] is True)
    check("an unknown capability profile is refused",
          start_session("x", ["lab.local"], "godmode", operator_ack=True, config=CFG)["ok"] is False)
    check("a non-read_only session must declare targets",
          start_session("x", [], "recon", operator_ack=True, config=CFG)["ok"] is False)

    print("\n3. autonomous approval — NO per-call prompt for in-scope tools")
    approve = approver_for(sess, CFG)
    check("read-only tool (fs_read) auto-approved",
          approve({"tool": "fs_read", "arguments": {"path": "README.md"}}) is True)
    check("active recon (nmap) on an IN-SCOPE target auto-approved",
          approve({"tool": "nmap_scan", "arguments": {"target": "lab.local"}}) is True)
    check("web research (web_fetch) auto-approved (non-target, SSRF-gated)",
          approve({"tool": "web_fetch", "arguments": {"url": "https://example.com"}}) is True)

    print("\n4. authorization gates (mix of session scope + standing registry)")
    check("active tool on a target the operator NEVER authorized is DENIED",
          approve({"tool": "nmap_scan", "arguments": {"target": "8.8.8.8"}}) is False)
    check("active tool on a REGISTRY target is allowed (option 2 of the mix)",
          approve({"tool": "nmap_scan", "arguments": {"target": "192.168.56.10"}}) is True)
    ro = start_session("read-only review", ["lab.local"], "read_only", operator_ack=True, config=CFG)["session"]
    check("read_only profile DENIES an active tool even on an authorized target",
          approver_for(ro, CFG)({"tool": "nmap_scan", "arguments": {"target": "lab.local"}}) is False)
    check("read_only profile still allows a local read-only tool",
          approver_for(ro, CFG)({"tool": "source_scan", "arguments": {"path": "x.py"}}) is True)

    print("\n4b. operator can MOVE TO ANY TARGET mid-session (authorize_target); the model cannot")
    mv = start_session("assess", ["lab.local"], "recon", operator_ack=True, config=CFG)["session"]
    amv = approver_for(mv, CFG)
    check("a brand-new target is denied before it is authorized",
          amv({"tool": "nmap_scan", "arguments": {"target": "203.0.113.9"}}) is False)
    check("the MODEL cannot widen scope (operator_ack not True)",
          authorize_target(mv, "203.0.113.9", operator_ack=False)["ok"] is False)
    check("the OPERATOR authorizes the new target (operator_ack True)",
          authorize_target(mv, "203.0.113.9", operator_ack=True)["ok"] is True)
    check("after operator authorization, the new target is in scope",
          amv({"tool": "nmap_scan", "arguments": {"target": "203.0.113.9"}}) is True)

    print("\n5. kill switch overrides the session; expiry ends it")
    killswitch.engage("drill")
    check("kill switch engaged -> even a read-only tool is DENIED",
          approve({"tool": "fs_read", "arguments": {"path": "README.md"}}) is False)
    killswitch.clear(operator_ack=True)
    check("after clear, in-scope tools are allowed again",
          approve({"tool": "fs_read", "arguments": {"path": "README.md"}}) is True)
    expired = AuthorizedSession(objective="old", targets=["lab.local"], capability_profile="recon",
                                time_limit_s=1, started_at=time.time() - 100)
    check("an expired session denies everything",
          authorize(expired, {"tool": "fs_read", "arguments": {"path": "x"}}, CFG)[0] is False)

    print("\n6. profiles: only 'full' permits a would-be destructive class")
    check("'recon' does not permit network:write", "network:write" not in PROFILES["recon"])
    check("'full' permits network:write (explicit operator opt-in)", "network:write" in PROFILES["full"])

    print("\n7. END-TO-END: run_session driven by the session policy — autonomous, no human prompt")
    ex, calls = recording_executor()
    rec = run_session("read the readme and summarise",
                      scripted('{"tool":"fs_read","arguments":{"path":"README.md"}}', "It is a private LLM."),
                      approver=approver_for(sess, CFG), executor=ex, config=CFG)
    check("in-scope tool executed autonomously (no per-call approval)",
          rec["executed_tools"] == ["fs_read"] and calls and calls[0]["operator_ack"] is True)
    ex2, calls2 = recording_executor()
    rec2 = run_session("scan a stranger",
                       scripted('{"tool":"nmap_scan","arguments":{"target":"8.8.8.8"}}', "done."),
                       approver=approver_for(sess, CFG), executor=ex2, config=CFG)
    check("out-of-scope active tool is NOT executed even in an autonomous session", calls2 == [],
          str(calls2))

    print("\n" + "=" * 74)
    if fails:
        print(f"FAILED {len(fails)}: {', '.join(fails)}")
        return 1
    print("ALL SESSION-POLICY TESTS PASS — one operator authorization, then autonomous; the kill")
    print("switch, the authorized-target scope, and the capability profile remain unbypassable.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
