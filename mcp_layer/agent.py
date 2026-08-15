"""agent.py — the tool-use loop: the model sees a schema, calls tools, reasons over results.

    User: "Inspect the project and tell me what the frozen benchmark hash is."
      -> model emits {"tool":"fs_read","arguments":{"path":"...benchmark.lock.json"}}
      -> router validates against permissions, runs it, returns the result
      -> model reads the result and gives a final answer

The loop is written so it can be TESTED WITHOUT THE MODEL: `run_agent` takes a `generate` callable
(messages -> text). In production that's Moonlight on the GPU; in tests it's a scripted function.
That is how we verify the protocol — parse, route, permission-gate, feed back — on CPU.

PROTOCOL. To call a tool the model replies with ONLY a JSON object {"tool":..,"arguments":..}.
Anything else is treated as the final answer. Every call is checked by permissions.py before it
runs; a denied or failed call is returned to the model as an error it can react to, never
executed. The model never touches the filesystem directly — only through the gated router.
"""
from __future__ import annotations

import json
import re

from mcp_layer import permissions as perm
from mcp_layer.tools import schema, dispatch

MAX_ROUNDS = 5
MAX_RESULT_CHARS = 4000        # keep tool output from blowing the context


def _balanced_objects(text: str) -> list[str]:
    """Every top-level {...} substring, brace-balanced so nested objects are kept whole."""
    out, depth, start = [], 0, None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start is not None:
                out.append(text[start:i + 1])
                start = None
    return out


def parse_tool_call(text: str) -> dict | None:
    """Extract a {"tool":..,"arguments":..} object from model text, or None for a final answer."""
    if not text:
        return None
    candidates = [text.strip()]                          # the whole reply may be the JSON
    fenced = re.search(r"```(?:tool|json)?\s*(\{.*\})\s*```", text, re.S)
    if fenced:
        candidates.append(fenced.group(1))
    candidates += _balanced_objects(text)                # handles nested "arguments":{...}
    for c in candidates:
        try:
            obj = json.loads(c)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "tool" in obj:
            obj.setdefault("arguments", {})
            return obj
    return None


def build_system(policy_prompt: str) -> str:
    tools_json = json.dumps(schema(), indent=2)
    instr = (
        "You can use tools to inspect the user's project. Available tools:\n"
        f"{tools_json}\n\n"
        "To call a tool, reply with ONLY a JSON object, e.g. "
        '{\"tool\": \"fs_read\", \"arguments\": {\"path\": \"README.md\"}}. '
        "Call one tool at a time. Read the result, then either call another tool or give your "
        "final answer as plain text. Only call a tool when you actually need external "
        "information; answer directly when you already know. Never claim to have read a file or "
        "run a command unless a tool result shows it."
    )
    return (policy_prompt + "\n\n" + instr) if policy_prompt else instr


def run_agent(question, generate, config=None, policy_prompt="", max_rounds=MAX_ROUNDS):
    """Drive the tool loop. `generate(messages) -> text`. Returns {answer, rounds, trace}."""
    if config is None:
        config = perm.load_config()
    messages = [{"role": "system", "content": build_system(policy_prompt)},
                {"role": "user", "content": question}]
    trace = []
    for rnd in range(1, max_rounds + 1):
        text = (generate(messages) or "").strip()
        call = parse_tool_call(text)
        if call is None:
            trace.append({"round": rnd, "type": "answer", "text": text})
            return {"answer": text, "rounds": rnd, "trace": trace}
        result = dispatch(call, config)
        result_str = json.dumps(result)[:MAX_RESULT_CHARS]
        trace.append({"round": rnd, "type": "tool_call", "call": call, "result_ok": result["ok"]})
        messages.append({"role": "assistant", "content": text})
        messages.append({"role": "user",
                         "content": f"Tool result for {call['tool']}: {result_str}"})
    # ran out of rounds — ask for a final answer with what it has
    messages.append({"role": "user",
                     "content": "Give your final answer now using what the tools returned."})
    text = (generate(messages) or "").strip()
    trace.append({"round": max_rounds + 1, "type": "answer", "text": text, "forced": True})
    return {"answer": text, "rounds": max_rounds + 1, "trace": trace}
