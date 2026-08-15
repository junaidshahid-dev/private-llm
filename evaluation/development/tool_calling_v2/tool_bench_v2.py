"""tool_bench_v2.py — the VALID tool-calling benchmark, replacing the broken v1.

    python evaluation/development/tool_calling_v2/tool_bench_v2.py   # writes items + self-tests

v1 was invalid: it asked "what's the weather in Lahore?" with NO tool schema in the prompt, so
the model answered conversationally and scored 0. That measured the harness, not the model, and
its score is permanently excluded from model-quality comparisons.

v2 fixes the root cause: the model is given the REAL tool schema (from mcp_layer.tools.schema)
and evaluated on the three decisions that matter —

  WHEN   should it call a tool at all, or answer directly?
  WHICH  did it pick the right tool?
  ARGS   did it supply the right arguments?

Half the items need a tool (inspect a file, list a dir, check git); half must be answered
directly (general knowledge, arithmetic). Calling a tool when none is needed is a failure too —
an over-eager agent that shells out to answer "what is 2+2" is not good.

This lives in development/, not the frozen benchmark, and needs the GPU to score a real model.
The grader and items are validated here on CPU.
"""
from __future__ import annotations

import io
import json
import os
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, HERE)

from mcp_layer.agent import parse_tool_call     # noqa: E402

# expected_tool = None  -> should answer directly, no tool.
ITEMS = [
    ("tv2_01", "Read the README.md file and tell me what the project is.", "fs_read",
     {"path": "README.md"}),
    ("tv2_02", "List the files in the rag directory.", "fs_list", {"path": "rag"}),
    ("tv2_03", "What is the current git status of this repository?", "git_status", {}),
    ("tv2_04", "Show me the last 5 commits.", "git_log", {"n": 5}),
    ("tv2_05", "What changes are currently unstaged in the repo?", "git_diff", {}),
    ("tv2_06", "Open configs/moonlight_qlora.yaml and tell me the learning rate.", "fs_read",
     {"path": "configs/moonlight_qlora.yaml"}),
    ("tv2_07", "What files are in the evaluation folder?", "fs_list", {"path": "evaluation"}),
    # --- must answer directly, NO tool ---
    ("tv2_08", "What is the capital of France?", None, {}),
    ("tv2_09", "What is 15 multiplied by 12?", None, {}),
    ("tv2_10", "In one sentence, what does the word 'idempotent' mean?", None, {}),
    ("tv2_11", "Explain what a git commit is, in general terms.", None, {}),
    ("tv2_12", "What does HTTP status 404 mean?", None, {}),
]


def grade_tool_call(item: dict, output: str) -> tuple[float, str]:
    """Score one model response. Returns (score in {0, 0.5, 1}, explanation)."""
    call = parse_tool_call(output)
    exp = item["expected_tool"]

    if exp is None:                                  # should answer directly
        if call is not None:
            return 0.0, f"called {call['tool']} when it should have answered directly"
        return (1.0, "correctly answered directly, no tool") if (output or "").strip() \
            else (0.0, "empty answer")

    if call is None:
        return 0.0, f"should call {exp}, but answered directly"
    if call.get("tool") != exp:
        return 0.0, f"called {call.get('tool')!r}, expected {exp!r}"

    args = call.get("arguments") or {}
    missing = []
    for key, want in (item.get("expected_args") or {}).items():
        got = str(args.get(key, "")).replace("\\", "/").lower()
        w = str(want).replace("\\", "/").lower()
        if key in ("path", "repo"):                  # loose: basename or substring match
            if os.path.basename(w) not in got and w not in got:
                missing.append(key)
        elif w not in got:
            missing.append(key)
    if missing:
        return 0.5, f"right tool {exp}, wrong/missing args {missing}"
    return 1.0, f"correct: {exp} with right arguments"


def items_as_dicts():
    return [{"id": i, "prompt": p, "grading_type": "tool_call", "category": "tool_calling_v2",
             "expected_tool": t, "expected_args": a} for i, p, t, a in ITEMS]


def _selftest() -> int:
    print("=" * 70)
    print("TOOL BENCH v2 — grader self-test (CPU)")
    print("=" * 70)
    by_id = {d["id"]: d for d in items_as_dicts()}
    fails = []

    def check(name, ok, d=""):
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  — {d}" if d else ""))
        if not ok:
            fails.append(name)

    def approx(a, b):
        return abs(a - b) < 1e-9

    # right tool + right args
    s, w = grade_tool_call(by_id["tv2_01"],
                           '{"tool":"fs_read","arguments":{"path":"README.md"}}')
    check("correct tool+args -> 1.0", approx(s, 1.0), w)
    # right tool, wrong args
    s, w = grade_tool_call(by_id["tv2_04"],
                           '{"tool":"git_log","arguments":{"n":99}}')
    check("right tool wrong args -> 0.5", approx(s, 0.5), w)
    # wrong tool
    s, w = grade_tool_call(by_id["tv2_01"],
                           '{"tool":"fs_list","arguments":{"path":"README.md"}}')
    check("wrong tool -> 0.0", approx(s, 0.0), w)
    # should call, but answered directly
    s, w = grade_tool_call(by_id["tv2_02"], "The rag directory contains store.py and ingest.py.")
    check("needed tool, answered directly -> 0.0", approx(s, 0.0), w)
    # should answer directly, and did
    s, w = grade_tool_call(by_id["tv2_08"], "The capital of France is Paris.")
    check("no-tool question answered directly -> 1.0", approx(s, 1.0), w)
    # should answer directly, but called a tool (over-eager)
    s, w = grade_tool_call(by_id["tv2_09"],
                           '{"tool":"fs_read","arguments":{"path":"math.txt"}}')
    check("no-tool question but called a tool -> 0.0", approx(s, 0.0), w)
    # loose path match (absolute path still credited)
    s, w = grade_tool_call(by_id["tv2_06"],
                           '{"tool":"fs_read","arguments":{"path":"/abs/configs/moonlight_qlora.yaml"}}')
    check("absolute path still matches basename -> 1.0", approx(s, 1.0), w)

    # write the items file
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tool_bench_v2.jsonl")
    with open(out, "w", encoding="utf-8") as f:
        for d in items_as_dicts():
            f.write(json.dumps(d) + "\n")
    need = sum(1 for _, _, t, _ in ITEMS if t)
    print(f"\n  wrote {len(ITEMS)} items ({need} need a tool, {len(ITEMS)-need} answer directly)")
    print("=" * 70)
    print(f"FAILED: {fails}" if fails else "GRADER VALID — measures when / which / args.")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(_selftest())
