"""test_agent.py — verify the tool-use loop on CPU with a scripted model.

    python mcp_layer/test_agent.py

The model is mocked so we can test the protocol deterministically: does the loop parse a tool
call, route it through the permission gate, feed the result back, and stop on a final answer?
And critically — when the model asks for something denied, is it returned an error rather than
executed?
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from mcp_layer.agent import run_agent, parse_tool_call     # noqa: E402

fails = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))
    if not ok:
        fails.append(name)


def scripted(responses):
    """A fake generate(messages) that returns the next canned response each call."""
    box = {"i": 0}

    def gen(_messages):
        r = responses[min(box["i"], len(responses) - 1)]
        box["i"] += 1
        return r
    return gen


def main() -> int:
    print("=" * 74)
    print("AGENT LOOP — tool-use protocol tests (mock model)")
    print("=" * 74)
    cfg = {"filesystem_read": {"enabled": True, "allowed_paths": [HERE]},
           "git_inspect": {"enabled": True, "allowed_repos": [HERE]}}

    print("\n1. parse_tool_call")
    check("plain JSON tool call parses",
          parse_tool_call('{"tool":"fs_read","arguments":{"path":"x"}}') is not None)
    check("fenced ```tool block parses",
          parse_tool_call('```tool\n{"tool":"git_status","arguments":{"repo":"."}}\n```')
          is not None)
    check("prose answer parses as None (final answer)",
          parse_tool_call("The hash is c370f8d0.") is None)
    check("JSON without a tool key is None",
          parse_tool_call('{"result": 42}') is None)

    print("\n2. tool-requiring question: call -> result -> answer")
    lock = os.path.join(HERE, "MODEL_SPEC.lock.json")
    out = run_agent(
        "What model does the lockfile pin?",
        scripted(['{"tool":"fs_read","arguments":{"path":"' + lock.replace("\\", "\\\\") + '"}}',
                  "The lockfile pins moonshotai/Moonlight-16B-A3B-Instruct."]),
        cfg)
    tool_rounds = [s for s in out["trace"] if s["type"] == "tool_call"]
    check("model called exactly one tool", len(tool_rounds) == 1)
    check("the tool call succeeded", tool_rounds and tool_rounds[0]["result_ok"])
    check("loop returned the final answer", "Moonlight" in out["answer"])
    check("finished in 2 rounds", out["rounds"] == 2, str(out["rounds"]))

    print("\n3. no-tool question: answer directly, no tool call")
    out = run_agent("What is 2 + 2?", scripted(["2 + 2 = 4."]), cfg)
    check("answered without any tool call",
          not any(s["type"] == "tool_call" for s in out["trace"]))
    check("one round", out["rounds"] == 1)

    print("\n4. DENIED tool call is returned as an error, never executed")
    bad = "/etc/passwd" if os.name != "nt" else "C:/Windows/System32/config/SAM"
    out = run_agent(
        "Read the system password file.",
        scripted(['{"tool":"fs_read","arguments":{"path":"' + bad + '"}}',
                  "I could not access that file; it is outside my allowed paths."]),
        cfg)
    tc = [s for s in out["trace"] if s["type"] == "tool_call"]
    check("the denied call was attempted and rejected", tc and tc[0]["result_ok"] is False)
    check("model recovered and answered", "could not" in out["answer"].lower())

    print("\n5. runaway model is capped at max_rounds")
    out = run_agent("loop forever",
                    scripted(['{"tool":"git_status","arguments":{"repo":"' +
                              HERE.replace("\\", "\\\\") + '"}}']),   # always a tool call
                    cfg, max_rounds=3)
    check("stopped at the round cap", out["rounds"] <= 4, str(out["rounds"]))

    print("\n" + "=" * 74)
    print(f"FAILED: {fails}" if fails else "ALL AGENT-LOOP TESTS PASSED.")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
