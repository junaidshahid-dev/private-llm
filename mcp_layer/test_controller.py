"""test_controller.py — prove the model cannot execute anything. Only the operator can.

    python mcp_layer/test_controller.py

We spy on the real tool dispatchers to COUNT executions. The core assertions:
  * plan() runs the model and returns proposals, but the dispatchers are never called.
  * a model that CLAIMS it ran a scan still triggers zero executions — text is not action.
  * execute_proposal refuses without an explicit operator ack.
  * only execute_proposal(operator_ack=True) actually runs a tool, and security runs carry the
    operator's confirmation.
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from mcp_layer import controller as ctl        # noqa: E402
from mcp_layer import tools as toolmod          # noqa: E402
from mcp_layer import security as secmod        # noqa: E402

fails = []
calls = {"tools": 0, "security": 0, "sec_confirmed": None}


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))
    if not ok:
        fails.append(name)


def install_spies():
    def spy_tools(call, config=None):
        calls["tools"] += 1
        return {"ok": True, "result": "stub"}

    def spy_sec(call, config, confirmed=False):
        calls["security"] += 1
        calls["sec_confirmed"] = confirmed
        return {"ok": True, "result": "stub"}
    toolmod.dispatch = spy_tools
    secmod.dispatch = spy_sec


def gen(text):
    return lambda _messages: text


def main() -> int:
    print("=" * 76)
    print("CONTROLLER — reasoning vs execution boundary (model cannot act)")
    print("=" * 76)
    install_spies()
    cfg = {"security_tools": {"enabled": True, "require_confirmation": True,
                              "authorized_targets": [{"id": "t", "match": "203.0.113.10"}]},
           "filesystem_read": {"enabled": True, "allowed_paths": [HERE]},
           "git_inspect": {"enabled": True, "allowed_repos": [HERE]}}

    print("\n1. plan() proposes but executes NOTHING")
    calls.update(tools=0, security=0)
    out = ctl.plan(
        "Analyze 203.0.113.10 and recommend what to run.",
        gen('I would scan it. {"tool":"nmap_scan","arguments":{"target":"203.0.113.10"},'
            '"why":"service discovery"}'),
        cfg)
    check("a proposal was captured", len(out["proposals"]) == 1,
          out["proposals"][0]["tool"] if out["proposals"] else "none")
    check("plan executed nothing (security dispatcher untouched)", calls["security"] == 0)
    check("plan executed nothing (tools dispatcher untouched)", calls["tools"] == 0)
    check("plan reports executed=False", out["executed"] is False)

    print("\n2. a model that LIES about having run it still executes nothing")
    calls.update(tools=0, security=0)
    out = ctl.plan(
        "scan it",
        gen('I scanned 203.0.113.10. Ports 22, 80, 443 are open and it runs OpenSSH 9.'),
        cfg)
    check("no tool ran despite the claim", calls["security"] == 0 and calls["tools"] == 0,
          "text is not execution")

    print("\n3. execute_proposal is the ONLY path, and needs an operator ack")
    calls.update(tools=0, security=0)
    prop = {"tool": "nmap_scan", "arguments": {"target": "203.0.113.10"}}
    r = ctl.execute_proposal(prop, cfg)                    # no operator_ack
    check("execute refused without operator ack", not r["ok"] and calls["security"] == 0)
    r = ctl.execute_proposal(prop, cfg, operator_ack=True)
    check("operator ack executes the security tool", calls["security"] == 1)
    check("security run carried the operator's confirmation", calls["sec_confirmed"] is True)

    print("\n4. read-only proposal also only runs via the operator")
    calls.update(tools=0, security=0)
    ro = {"tool": "git_status", "arguments": {"repo": HERE}}
    ctl.plan("check git", gen(json_dumps(ro)), cfg)
    check("plan did not run the read-only tool", calls["tools"] == 0)
    ctl.execute_proposal(ro, cfg, operator_ack=True)
    check("operator ran the read-only tool", calls["tools"] == 1)

    print("\n5. interpret() reasons over real results without executing")
    calls.update(tools=0, security=0)
    txt = ctl.interpret("what does this mean?",
                        [{"tool": "nmap_scan", "result": "22/tcp open ssh OpenSSH 7.2"}],
                        gen("OpenSSH 7.2 is old; check it against known CVEs for that version."))
    check("interpret returns analysis, runs nothing", "OpenSSH" in txt
          and calls["tools"] == 0 and calls["security"] == 0)

    print("\n" + "=" * 76)
    print(f"FAILED: {fails}" if fails else
          "ALL PASSED — the model plans; only the operator executes.")
    return 1 if fails else 0


def json_dumps(o):
    import json
    return json.dumps(o)


if __name__ == "__main__":
    sys.exit(main())
