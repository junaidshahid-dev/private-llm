"""capture_recon_test.py — the capture structures real command output like a tool result.

    python bridge/capture_recon_test.py

Runs harmless local commands (no Docker, no lab needed) and checks the shape matches what
controller.interpret / the MCP loop expect: {"tool", "result": {"ok", "command", "output"}}.
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

from bridge.capture_recon import run_step, capture                           # noqa: E402

fails = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))
    if not ok:
        fails.append(name)


def main() -> int:
    print("=" * 70)
    print("CAPTURE TEST — real command output structured as a tool result")
    print("=" * 70)

    ok_step = run_step("probe", [sys.executable, "--version"])
    check("successful command => ok True", ok_step["result"]["ok"])
    check("captures the command", "python" in ok_step["result"]["command"].lower())
    check("captures real output", "python" in (ok_step["result"]["output"] or "").lower(),
          ok_step["result"].get("output"))
    check("shape matches the loop's {tool, result}", set(ok_step) == {"tool", "result"})

    bad = run_step("nope", ["this_binary_does_not_exist_xyz"])
    check("missing binary => ok False + error", not bad["result"]["ok"] and "error" in bad["result"])

    cap = capture(steps=[("probe", [sys.executable, "-c", "print('recon evidence 12345')"])])
    check("capture has target/task/results", {"target", "task", "results"} <= set(cap))
    check("results carry the real output",
          "recon evidence 12345" in cap["results"][0]["result"]["output"])
    check("task tells the model to separate evidence from inference",
          "evidence" in cap["task"] and "inference" in cap["task"])

    # controller.interpret must accept this shape without error (JSON-serializable, evidence intact)
    import json
    dumped = json.dumps(cap["results"])
    check("results are JSON-serializable for interpret()", "recon evidence 12345" in dumped)

    print("\n" + "=" * 70)
    if fails:
        print(f"FAILED {len(fails)}: {', '.join(fails)}")
        return 1
    print("ALL CAPTURE TESTS PASS — real tool output is captured in the loop's format.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
