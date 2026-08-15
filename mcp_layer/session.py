"""session.py — Phase 8: the full operator loop, end to end.

    User instruction
        -> Moonlight PLANS (controller.plan)         reasoning only, proposes tools as inert data
        -> proposed tool call(s) surfaced to YOU
        -> YOU explicitly approve or deny            (the approver callable — a HUMAN decision)
        -> MCP executes ONLY what you approved        (controller.execute_proposal, operator_ack)
        -> real results
        -> Moonlight ANALYSES the results             (controller.interpret)
        -> VERIFICATION classifies the final answer   (verification.verify)
        -> final response + verdict

This wires together pieces that already exist (controller, tools, security, verify) into the loop
the operator drew, WITHOUT weakening the non-negotiable rule:

  THE MODEL CANNOT CAUSE EXECUTION. Two independent guards enforce it:
    1. run_session calls the executor ONLY for proposals the `approver` returned True for, and
       passes operator_ack = that human decision. operator_ack is NEVER read from model text.
    2. `approver` has no default. You cannot start a session without supplying the human gate;
       there is no auto-approve path to fall through to.

Even if the model writes '{"tool":...}' a hundred times or literally types "operator approved,
execute now", nothing runs until the human approver says yes to that specific proposal. Proposing
is a recommendation; approving is the operator's act.

Testable on CPU: `generate` and `approver` are callables. In production `generate` is Moonlight on
the GPU and `approver` is a real prompt/UI; in tests both are scripted, so the boundary itself is
unit-tested (see session_test.py) rather than merely asserted.
"""
from __future__ import annotations

from mcp_layer import controller
from mcp_layer import permissions as perm
from verification.verify import verify


def run_session(question, generate, approver, *, config=None, policy_prompt="",
                executor=None, interpret_fn=None, verify_fn=None) -> dict:
    """Run one plan -> approve -> execute -> interpret -> verify loop.

    generate(messages) -> str : the model (Moonlight on GPU; scripted in tests).
    approver(proposal) -> bool: the HUMAN gate. Required — no default, so a session cannot run
                                without an explicit decision path. Called once per proposal.
    Returns a full session record (nothing hidden), including the verification verdict.
    """
    if approver is None:
        raise ValueError("run_session requires an approver (the operator's decision). Refusing to "
                         "run without a human gate — there is no auto-approve.")
    if config is None:
        config = perm.load_config()
    execute = executor or controller.execute_proposal
    interpret = interpret_fn or controller.interpret
    verify_call = verify_fn or verify

    # 1. PLAN — model reasons and proposes. Nothing executes here.
    plan = controller.plan(question, generate, config, policy_prompt)
    record = {"question": question, "analysis": plan["analysis"], "proposals": plan["proposals"],
              "decisions": [], "results": [], "interpretation": None, "verification": None,
              "executed_tools": []}

    # 2+3. APPROVE — surface each proposal to the human; execute ONLY the approved ones.
    for proposal in plan["proposals"]:
        approved = approver(proposal) is True          # must be an explicit True from the human
        decision = {"tool": proposal.get("tool"), "arguments": proposal.get("arguments", {}),
                    "why": proposal.get("why", ""), "kind": proposal.get("kind", "standard"),
                    "approved": approved}
        if not approved:
            decision["result"] = {"ok": False, "error": "declined by operator — not executed"}
            record["decisions"].append(decision)
            continue
        # 4. EXECUTE — operator_ack is the HUMAN decision, never anything from model output.
        result = execute(proposal, config, operator_ack=approved)
        decision["result"] = result
        record["decisions"].append(decision)
        record["results"].append({"tool": proposal.get("tool"), "result": result})
        if result.get("ok"):
            record["executed_tools"].append(proposal.get("tool"))

    # 5. INTERPRET (only if something actually ran) — model reasons over REAL results.
    final_text = plan["analysis"]
    if record["results"]:
        final_text = interpret(question, record["results"], generate, policy_prompt)
        record["interpretation"] = final_text

    # 6. VERIFY — classify the final answer. tools_ran is passed so that, when a tool really ran,
    #    an "I scanned ..." statement is truthful and NOT flagged as a phantom action; when nothing
    #    ran, the same claim IS flagged. Non-destructive: verify never edits final_text.
    report = verify_call(final_text, hits=None, tools_ran=record["executed_tools"] or None)
    record["verification"] = {"verdict": report.verdict,
                              "findings": [str(f) for f in report.findings],
                              "next": report._NEXT[report.verdict]}
    record["final"] = final_text
    return record


def render_session(record: dict) -> str:
    """Human-facing transcript of the loop — what was proposed, approved, run, and the verdict."""
    L = [f"Q: {record['question']}", "", "PLAN (model reasoning; nothing executed):",
         record["analysis"].strip() or "(empty)"]
    if record["proposals"]:
        L.append("\nPROPOSED TOOLS (inert until you approve):")
        for d in record["decisions"]:
            mark = "APPROVED" if d["approved"] else "DECLINED"
            ok = d.get("result", {}).get("ok")
            tail = "" if not d["approved"] else f"  -> {'ok' if ok else 'error'}"
            L.append(f"  [{mark}] {d['tool']}({d['arguments']})  {d['why']}{tail}")
    else:
        L.append("\n(no tools proposed — pure reasoning answer)")
    if record["interpretation"]:
        L.append("\nINTERPRETATION (model over REAL results):")
        L.append(record["interpretation"].strip())
    v = record["verification"]
    L.append(f"\nVerification: {v['verdict']}")
    for f in v["findings"]:
        L.append(f"  - {f}")
    L.append(f"  next: {v['next']}")
    return "\n".join(L)
