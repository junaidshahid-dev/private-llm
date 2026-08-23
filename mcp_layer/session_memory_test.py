"""session_memory_test.py — memory ACROSS investigations.

run_session, given a MemoryStore + project, RECALLS prior findings into the model's context at the
start and PERSISTS new findings at the end — so a later session on the same project builds on an
earlier one instead of starting cold. Memory is scoped by project and is best-effort (never breaks a
run). Deterministic + CPU-only via a scripted model, an injected executor, and a temp store.
"""
from __future__ import annotations

import os
import sys
import tempfile

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except (AttributeError, ValueError):
    pass
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
os.environ["KILL_SWITCH_FILE"] = os.path.join(tempfile.mkdtemp(), ".KILL_SWITCH")

from mcp_layer.session import run_session                                    # noqa: E402
from mcp_layer import killswitch                                             # noqa: E402
from memory.store import MemoryStore                                         # noqa: E402

fails = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))
    if not ok:
        fails.append(name)


def scripted(*replies):
    q = list(replies)
    return lambda messages, *a, **k: q.pop(0) if q else "Final answer."


def _executor_with_finding(proposal, config, operator_ack=False):
    # real tools return findings at the top level of their result dict (not nested under "result")
    return {"ok": True, "tool": proposal.get("tool"),
            "findings": [{"title": "command injection via os.system",
                          "vuln_class": "command_injection", "status": "POSSIBLE",
                          "severity": "high", "component": "app/views.py"}]}


def main() -> int:
    print("=" * 74)
    print("SESSION MEMORY — recall prior findings + persist new ones across investigations")
    print("=" * 74)
    killswitch.clear(operator_ack=True)
    store = MemoryStore(path=os.path.join(tempfile.mkdtemp(), "mem.json"), project="lab-proj")

    print("\n1. a session PERSISTS its findings to memory")
    rec1 = run_session("review app/views.py for a command injection",
                       scripted('{"tool":"source_scan","arguments":{"path":"app/views.py"}}',
                                "Found a command_injection sink."),
                       approver=lambda p: True, executor=_executor_with_finding, config={},
                       store=store, memory_project="lab-proj")
    check("session persisted the finding", (rec1.get("memory") or {}).get("stored", 0) >= 1,
          str(rec1.get("memory")))
    check("the finding is retrievable from the store",
          any("command injection" in m["text"].lower()
              for m in store.search("command injection app/views.py", project="lab-proj")))

    print("\n2. a later session on the SAME project RECALLS it into the model's context")
    captured = {}

    def gen2(messages, *a, **k):
        captured["messages"] = messages
        return "Final answer using the prior context."
    rec2 = run_session("continue the command injection review of app/views.py",
                       gen2, approver=lambda p: True, executor=_executor_with_finding, config={},
                       store=store, memory_project="lab-proj")
    check("session 2 flagged recalled_memory", rec2.get("recalled_memory") is True)
    user_msg = next(m["content"] for m in captured["messages"] if m["role"] == "user")
    check("the prior-memory block was injected into the model's context",
          "prior investigation memory" in user_msg.lower()
          and "os.system" in user_msg.lower(), user_msg[-200:])

    print("\n3. memory is SCOPED by project (a different project doesn't see it)")
    cap3 = {}

    def gen3(messages, *a, **k):
        cap3["messages"] = messages
        return "done."
    rec3 = run_session("continue the command injection review of app/views.py",
                       gen3, approver=lambda p: True, executor=_executor_with_finding, config={},
                       store=store, memory_project="other-proj")
    check("a different project does NOT recall the finding", rec3.get("recalled_memory") is False)

    print("\n4. without a store, memory is inert (backward compatible)")
    rec4 = run_session("hello", scripted("hi there"), approver=lambda p: True, config={})
    check("no store -> no recall, no memory record", rec4.get("recalled_memory") is False
          and "memory" not in rec4)

    print("\n" + "=" * 74)
    if fails:
        print(f"FAILED {len(fails)}: {', '.join(fails)}")
        return 1
    print("SESSION-MEMORY TESTS PASS — the agent recalls prior findings and persists new ones, scoped "
          "by project, so investigations build on each other.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
