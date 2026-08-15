"""controller.py — the hard boundary between REASONING and EXECUTION.

Non-negotiable requirement, enforced architecturally (not by a system prompt): the model's output
cannot trigger a tool. The model reasons and PROPOSES; the operator EXECUTES. There is no code
path from generate() to a side-effecting tool.

  plan(question, generate, config)
      Drives the model in analysis mode. Returns its analysis and any proposed tool calls as
      INERT DATA. Executes nothing. Even if the model writes "I ran the scan, here are the
      results", no tool ran — the model only produced text, and text is not execution.

  execute_proposal(proposal, config, operator_ack=True)
      The OPERATOR's execution path. Runs exactly one approved proposal, and only when
      operator_ack is explicitly True — a value the operator's harness supplies, never derivable
      from model output. Security tools are routed with confirmed=True because reaching this
      function already IS the operator's explicit instruction.

  interpret(question, results, generate, config)
      After the operator runs something, feed the real results back for the model to interpret
      and correlate. This is how the model reasons over tool output without ever having run it.

The three functions are separate on purpose. The model can only ever influence plan() and
interpret() (both text-only). Execution lives behind execute_proposal(), which the model has no
way to call. That is the architectural guarantee: model instructions are not enough to act.
"""
from __future__ import annotations

import json

from mcp_layer import tools as toolmod
from mcp_layer import security as secmod
from mcp_layer import permissions as perm
from mcp_layer.agent import _balanced_objects

REASONING_SYSTEM = (
    "You are in ANALYSIS mode, assisting an operator who runs every action themselves.\n"
    "You may analyse the environment, ask for missing details, form hypotheses, recommend which "
    "tools fit, interpret results the operator gives you, and draw precise conclusions.\n"
    "You do NOT run anything. If a tool would help, PROPOSE it as a JSON object "
    '{\"tool\": name, \"arguments\": {...}, \"why\": \"...\"}. Proposing is a recommendation, not '
    "an action — the operator decides whether to execute it.\n"
    "Never claim you have run a tool, scanned a target, read a file, or seen output unless a tool "
    "result is actually provided to you. If you have no results yet, say what you would run and "
    "why, and wait."
)


def _all_tool_names() -> set:
    return set(toolmod.DISPATCH) | set(secmod.DISPATCH)


def parse_proposals(text: str) -> list[dict]:
    """Every {"tool":..} object in the model's text, as inert proposals. Executes nothing."""
    names = _all_tool_names()
    out, seen = [], set()
    for chunk in _balanced_objects(text or ""):
        try:
            obj = json.loads(chunk)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and obj.get("tool") in names:
            key = json.dumps({"t": obj["tool"], "a": obj.get("arguments", {})}, sort_keys=True)
            if key not in seen:
                seen.add(key)
                out.append({"tool": obj["tool"], "arguments": obj.get("arguments", {}) or {},
                            "why": obj.get("why", ""), "kind":
                            "security" if obj["tool"] in secmod.DISPATCH else "standard"})
    return out


def plan(question, generate, config=None, policy_prompt="") -> dict:
    """Model analyses and proposes. NOTHING is executed here."""
    system = (policy_prompt + "\n\n" + REASONING_SYSTEM) if policy_prompt else REASONING_SYSTEM
    messages = [{"role": "system", "content": system}, {"role": "user", "content": question}]
    analysis = (generate(messages) or "").strip()
    return {"analysis": analysis, "proposals": parse_proposals(analysis), "executed": False}


def execute_proposal(proposal: dict, config=None, operator_ack: bool = False) -> dict:
    """OPERATOR-only. Runs one approved proposal. Refuses without an explicit operator ack."""
    if operator_ack is not True:
        return {"ok": False, "error": "execution requires explicit operator acknowledgement; "
                "the model cannot trigger this"}
    if config is None:
        config = perm.load_config()
    tool = (proposal or {}).get("tool")
    if tool in secmod.DISPATCH:
        # reaching here IS the operator's explicit instruction, so confirm the security run
        return secmod.dispatch(proposal, config, confirmed=True)
    if tool in toolmod.DISPATCH:
        return toolmod.dispatch(proposal, config)
    return {"ok": False, "error": f"unknown tool {tool!r}"}


def interpret(question, results, generate, policy_prompt="") -> str:
    """Feed real tool results back for the model to interpret and correlate."""
    sysmsg = (policy_prompt + "\n\n" + REASONING_SYSTEM) if policy_prompt else REASONING_SYSTEM
    blocks = "\n\n".join(f"[{i+1}] {r.get('tool','tool')} -> {json.dumps(r.get('result', r))[:2000]}"
                         for i, r in enumerate(results))
    messages = [{"role": "system", "content": sysmsg},
                {"role": "user", "content":
                 f"{question}\n\nThe operator ran these and gave you the results:\n{blocks}\n\n"
                 "Interpret them, correlate across sources, and give precise conclusions."}]
    return (generate(messages) or "").strip()
