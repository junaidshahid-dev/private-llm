"""killswitch_test.py — the global STOP is independent, overriding, and the model cannot lift it.

    python mcp_layer/killswitch_test.py
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

# isolate the kill file to a temp path so the test never touches the repo-root switch
_TMP = tempfile.mkdtemp()
os.environ["KILL_SWITCH_FILE"] = os.path.join(_TMP, ".KILL_SWITCH")

from mcp_layer import killswitch                                             # noqa: E402
from mcp_layer import controller                                            # noqa: E402

fails = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))
    if not ok:
        fails.append(name)


def main() -> int:
    print("=" * 74)
    print("KILL SWITCH — independent global STOP the model cannot lift")
    print("=" * 74)
    killswitch.clear(operator_ack=True)          # ensure a clean start

    print("\n1. engage / status / is_engaged")
    check("starts disengaged", killswitch.is_engaged() is False)
    check("guard is inert when clear", killswitch.guard("nmap_scan", "lab") is None)
    r = killswitch.engage("operator pulled the cord", by="test")
    check("engage succeeds", r.get("ok") and r.get("engaged"))
    check("is_engaged True after engage", killswitch.is_engaged() is True)
    st = killswitch.status()
    check("status carries reason + engaged", st["engaged"] and "cord" in st["reason"], str(st))

    print("\n2. guard BLOCKS when engaged")
    g = killswitch.guard("nmap_scan", "192.168.56.10")
    check("guard returns a blocked response", g is not None and g.get("blocked") is True)
    check("blocked response explains the halt", "KILL SWITCH ENGAGED" in g.get("error", ""))

    print("\n3. the model CANNOT clear it (only the operator boolean True)")
    check("clear('approved') refused (model text is not authorization)",
          killswitch.clear("approved").get("ok") is not True)
    check("clear(1) refused (truthy is not the boolean True)",
          killswitch.clear(1).get("ok") is not True)
    check("clear(True) refused? no — operator ack works", killswitch.clear(True).get("ok") is True)
    check("cleared -> disengaged", killswitch.is_engaged() is False)

    print("\n4. the EXECUTOR is overridden by the switch, even with a valid operator_ack")
    proposal = {"tool": "nmap_scan", "arguments": {"target": "192.168.56.10"}, "why": "recon"}
    killswitch.engage("halt for test")
    res = controller.execute_proposal(proposal, operator_ack=True)   # valid ack, but switch engaged
    check("execute_proposal is BLOCKED while engaged despite operator_ack=True",
          res.get("blocked") is True and "KILL SWITCH" in res.get("error", ""))
    killswitch.clear(operator_ack=True)
    # once clear, the normal gates apply again (a read-only tool with no ack is refused by the ack gate)
    res2 = controller.execute_proposal(proposal, operator_ack=False)
    check("after clear, the ordinary ack gate is back (unacked call refused, not kill-blocked)",
          res2.get("blocked") is not True and "acknowledge" in res2.get("error", "").lower())

    print("\n5. best-effort termination of a registered running process")
    class FakeProc:
        def __init__(self):
            self.terminated = False

        def terminate(self):
            self.terminated = True
    p = FakeProc()
    killswitch.register("job1", p)
    out = killswitch.engage("stop with a live job")
    check("engage terminated the registered process", p.terminated is True)
    check("engage reports how many it terminated", out.get("terminated_active") == 1)
    killswitch.clear(operator_ack=True)

    print("\n6. no model-facing tool can touch the switch")
    check("no 'kill'/'stop' tool in the tool surface",
          not any("kill" in n or "stop" in n for n in controller._all_tool_names()))

    print("\n" + "=" * 74)
    if fails:
        print(f"FAILED {len(fails)}: {', '.join(fails)}")
        return 1
    print("ALL KILL-SWITCH TESTS PASS — engaged state halts the executor independently of the model,")
    print("and only the operator's explicit acknowledgement can clear it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
