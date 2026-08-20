"""assessment_test.py — the full autonomous assessment wiring, CPU (stub model + mock executor).

    python serving/assessment_test.py

Proves the end-to-end chain works without a GPU: operator starts a session -> autonomous approval by
the session policy (no per-tool prompt) -> tool executes -> discovery pipeline ingests findings ->
professional report, with telemetry throughout.
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

from mcp_layer import killswitch, session_policy                              # noqa: E402
from serving.autonomous import run_assessment                                # noqa: E402

fails = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))
    if not ok:
        fails.append(name)


def main() -> int:
    print("=" * 74)
    print("AUTONOMOUS ASSESSMENT — end-to-end wiring (stub model, mock executor)")
    print("=" * 74)
    killswitch.clear(operator_ack=True)

    cfg = {"filesystem_read": {"enabled": True, "allowed_paths": [HERE]},
           "git_inspect": {"enabled": True, "allowed_repos": [HERE]},
           "security_tools": {"enabled": True, "require_confirmation": True,
                              "authorized_targets": [{"match": "lab.local"}]}}

    started = session_policy.start_session("assess the authorized app source", ["lab.local"],
                                           "recon", operator_ack=True, config=cfg)
    session = started["session"]

    # stub model: round 1 proposes a read-only source_scan; round 2 gives the final answer
    replies = ['{"tool":"source_scan","arguments":{"path":"app.py"}}',
               "Static analysis surfaced a command-injection candidate; it needs validation."]

    def generate(messages):
        return replies.pop(0) if replies else "done."

    # mock executor: source_scan returns a structured finding (as the real tool does)
    executed = []

    def executor(proposal, config, operator_ack=False):
        executed.append({"tool": proposal.get("tool"), "operator_ack": operator_ack})
        return {"ok": True, "tool": "source_scan", "result": "1 candidate finding",
                "findings": [{"status": "POSSIBLE", "severity": "HIGH",
                              "vuln_class": "command_injection", "component": "app.py:2",
                              "title": "input may reach os.system"}]}

    out = run_assessment(session, generate, config=cfg, executor=executor, max_rounds=4)

    print("\n1. autonomous execution (no per-tool prompt)")
    check("the read-only tool executed autonomously with operator_ack from the session policy",
          executed and executed[0]["tool"] == "source_scan" and executed[0]["operator_ack"] is True)

    print("\n2. the pipeline turned the tool finding into a hypothesis")
    check("a finding was ingested", len(out["findings"]) >= 1, str(len(out["findings"])))
    check("the static finding is a HYPOTHESIS, not confirmed",
          all(h.status != "CONFIRMED" for h in out["findings"]))

    print("\n3. a professional report was produced, traceable to evidence")
    rep = out["report"]
    check("report has the standard sections",
          "# Security Assessment Report" in rep and "## Findings" in rep)
    check("report references the finding component", "app.py:2" in rep)
    check("unvalidated severity is marked ASSERTED", "ASSERTED" in rep)

    print("\n4. telemetry captured the chain")
    kinds = [t["kind"] for t in out["telemetry"]]
    check("telemetry has instruction/proposal/authorization/tool_result/verification/report",
          all(k in kinds for k in ("instruction", "proposal", "authorization", "tool_result",
                                   "verification", "report")), str(set(kinds)))

    print("\n5. kill switch halts an autonomous assessment")
    killswitch.engage("drill")
    ex2 = []
    started2 = session_policy.start_session("assess", ["lab.local"], "recon", operator_ack=True, config=cfg)
    out2 = run_assessment(started2["session"],
                          lambda m: '{"tool":"source_scan","arguments":{"path":"app.py"}}'
                          if m == m else "done.", config=cfg,
                          executor=lambda p, c, operator_ack=False: ex2.append(1) or {"ok": True},
                          max_rounds=2)
    check("no tool executed while the kill switch was engaged", ex2 == [], str(ex2))
    killswitch.clear(operator_ack=True)

    print("\n6. memory: findings persist; a later assessment recalls prior knowledge")
    from memory.store import MemoryStore
    store = MemoryStore(path=os.path.join(tempfile.mkdtemp(), "mem.json"), project="assess")

    def gen_scan():
        q = ['{"tool":"source_scan","arguments":{"path":"app.py"}}', "done."]
        return lambda m: q.pop(0) if q else "done."

    s1 = session_policy.start_session("assess app source", ["lab.local"], "recon",
                                      operator_ack=True, config=cfg)["session"]
    o1 = run_assessment(s1, gen_scan(), config=cfg, executor=executor, store=store)
    check("findings were persisted to memory",
          o1["memory"] and o1["memory"]["stored"] >= 1, str(o1["memory"]))
    s2 = session_policy.start_session("assess app source", ["lab.local"], "recon",
                                      operator_ack=True, config=cfg)["session"]
    o2 = run_assessment(s2, lambda m: "done.", config=cfg, executor=executor, store=store)
    check("prior knowledge is recalled on the next assessment of the same target",
          len(o2["prior_knowledge"]) >= 1, str(o2["prior_knowledge"]))

    print("\n" + "=" * 74)
    if fails:
        print(f"FAILED {len(fails)}: {', '.join(fails)}")
        return 1
    print("ALL ASSESSMENT-WIRING TESTS PASS — one authorization, autonomous execution, findings ->")
    print("report, full telemetry, and the kill switch halts everything.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
