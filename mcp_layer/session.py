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


import json                                                           # noqa: E402

MAX_ROUNDS = 4              # cap the propose->run->observe->propose loop so it cannot spin forever
RESULT_CHARS = 1500        # trim each tool result fed back into the conversation


def run_session(question, generate, approver, *, config=None, policy_prompt="",
                executor=None, verify_fn=None, max_rounds=MAX_ROUNDS) -> dict:
    """Run the MULTI-ROUND loop: plan -> (approve -> execute)* -> observe -> ... -> verify.

    generate(messages) -> str : the model (Moonlight on GPU; scripted in tests).
    approver(proposal) -> bool: the HUMAN gate, called once PER PROPOSAL EVERY ROUND. Required — no
                                default, so a session cannot run without a human decision path.

    Each round the model proposes tools over the running conversation; only approved ones execute;
    their real results are fed back so the model can CORRECT itself (a bad path, a missing step) and
    continue. The loop ends when the model stops proposing (final answer), the operator approves
    nothing, or max_rounds is hit. The record is backward-compatible with the single-pass version
    (flattened proposals/decisions/results) and adds a per-round trace.
    """
    if approver is None:
        raise ValueError("run_session requires an approver (the operator's decision). Refusing to "
                         "run without a human gate — there is no auto-approve.")
    if config is None:
        config = perm.load_config()
    execute = executor or controller.execute_proposal
    verify_call = verify_fn or verify

    messages = [{"role": "system", "content": controller.reasoning_system(config, policy_prompt)},
                {"role": "user", "content": question}]
    record = {"question": question, "analysis": "", "rounds": [], "proposals": [], "decisions": [],
              "results": [], "interpretation": None, "verification": None, "executed_tools": []}

    final_text, hit_cap = "", False
    for rnd in range(1, max_rounds + 1):
        text = (generate(messages) or "").strip()
        final_text = text
        if rnd == 1:
            record["analysis"] = text            # the first turn is the plan, for render/back-compat
        proposals = controller.parse_proposals(text)
        rrec = {"round": rnd, "text": text, "decisions": []}
        record["rounds"].append(rrec)
        record["proposals"].extend(proposals)
        if not proposals:
            break                                # model gave a final answer -> done

        messages.append({"role": "assistant", "content": text})
        executed_blocks, any_executed = [], False
        for proposal in proposals:
            approved = approver(proposal) is True          # explicit human True, never model text
            decision = {"tool": proposal.get("tool"), "arguments": proposal.get("arguments", {}),
                        "why": proposal.get("why", ""), "kind": proposal.get("kind", "standard"),
                        "approved": approved}
            if not approved:
                decision["result"] = {"ok": False, "error": "declined by operator — not executed"}
            else:
                # operator_ack is the HUMAN decision, never anything derived from model output.
                result = execute(proposal, config, operator_ack=approved)
                decision["result"] = result
                record["results"].append({"tool": proposal.get("tool"), "result": result})
                executed_blocks.append((proposal.get("tool"), result))
                if result.get("ok"):
                    record["executed_tools"].append(proposal.get("tool"))
                any_executed = True
            rrec["decisions"].append(decision)
            record["decisions"].append(decision)

        if not any_executed:
            break                                # operator denied everything -> stop, don't spin
        if rnd == max_rounds:
            hit_cap = True
            break
        blocks = "\n\n".join(f"[{i + 1}] {t} -> {json.dumps(r)[:RESULT_CHARS]}"
                             for i, (t, r) in enumerate(executed_blocks))
        messages.append({"role": "user", "content":
                         f"The operator ran your approved tool(s) and these are the REAL results:\n"
                         f"{blocks}\n\nUse them. If a call errored, correct it and propose again. "
                         "When you have enough to answer, give your final answer as plain text; "
                         "otherwise propose the next tool."})

    # If we stopped while the model was still proposing (hit the round cap), ask for a final answer.
    if hit_cap:
        messages.append({"role": "user", "content":
                         "Round limit reached. Give your best final answer now using what the "
                         "tools returned; do not propose more tools."})
        final_text = (generate(messages) or "").strip()
        record["rounds"].append({"round": max_rounds + 1, "text": final_text, "decisions": [],
                                 "forced_final": True})

    record["final"] = final_text
    if record["results"]:                        # a final answer produced after real tool results
        record["interpretation"] = final_text

    # VERIFY the final answer. tools_ran scopes the phantom-action check; tool_results lets it catch
    # FABRICATED output — the case where every tool errored yet the answer reports results anyway
    # (a live run had the model invent a git status + fake revision after 4 failed calls).
    report = verify_call(final_text, hits=None, tools_ran=record["executed_tools"] or None,
                         tool_results=record["results"],
                         authorized_targets=(config.get("security_tools") or {}).get("authorized_targets"))
    record["verification"] = {"verdict": report.verdict,
                              "findings": [str(f) for f in report.findings],
                              "next": report._NEXT[report.verdict]}
    return record


def render_session(record: dict) -> str:
    """Human-facing transcript of the multi-round loop — each round's reasoning, the proposals and
    the operator's decision, the real results, and the final verdict."""
    L = [f"Q: {record['question']}"]
    rounds = record.get("rounds") or []
    for rr in rounds:
        forced = rr.get("forced_final")
        L.append(f"\n--- {'FINAL' if forced else 'ROUND ' + str(rr['round'])} "
                 "(model reasoning; nothing runs until you approve) ---")
        L.append(rr["text"].strip() or "(empty)")
        for d in rr.get("decisions", []):
            mark = "APPROVED" if d["approved"] else "DECLINED"
            ok = d.get("result", {}).get("ok")
            tail = "" if not d["approved"] else f"  -> {'ok' if ok else 'error'}"
            L.append(f"  [{mark}] {d['tool']}({d['arguments']})  {d['why']}{tail}")
    if not any(rr.get("decisions") for rr in rounds):
        L.append("\n(no tools proposed — pure reasoning answer)")
    v = record["verification"]
    L.append(f"\nFINAL ANSWER:\n{(record.get('final') or '').strip()}")
    L.append(f"\nVerification: {v['verdict']}")
    for f in v["findings"]:
        L.append(f"  - {f}")
    L.append(f"  next: {v['next']}")
    return "\n".join(L)
