"""tools.py — the read-only tool surface the model can call, gated by permissions.py.

    from mcp_layer.tools import schema, dispatch
    tools = schema()                      # show this to the model
    result = dispatch(call, config)       # run a validated tool call

Only read-only tools are implemented here: list/read files inside allowed paths, and inspect git
(status, log, diff, show) on allowed repos. Every call goes through permissions.py first; a denied
call returns {"ok": false, "error": ...} and executes nothing. Write and terminal are not here at
all yet — the safest way to not run a dangerous action is to not implement it.

The model's job: read the schema, emit {"tool": name, "arguments": {...}}, receive the result,
reason over it. That is also what the tool_calling benchmark was missing — real definitions.
"""
from __future__ import annotations

import os
import subprocess

from mcp_layer import permissions as perm

MAX_READ_BYTES = 100_000        # don't hand the model a 50MB file
GIT_TIMEOUT = 15


def schema() -> list[dict]:
    """Tool definitions to put in the model's prompt."""
    return [
        {"name": "fs_list", "description": "List the entries in a directory (read-only).",
         "arguments": {"path": "directory path inside an allowed root"}},
        {"name": "fs_read", "description": "Read a UTF-8 text file (read-only, truncated).",
         "arguments": {"path": "file path inside an allowed root"}},
        {"name": "git_status", "description": "Short git status of a repo.",
         "arguments": {"repo": "path to an allowed git repo"}},
        {"name": "git_log", "description": "Recent commits, one line each.",
         "arguments": {"repo": "allowed repo", "n": "how many (default 10, max 50)"}},
        {"name": "git_diff", "description": "Unstaged git diff of a repo.",
         "arguments": {"repo": "allowed repo"}},
    ]


def _run_git(repo: str, args: list[str]) -> dict:
    try:
        r = subprocess.run(["git", "-C", repo, *args], capture_output=True, text=True,
                           timeout=GIT_TIMEOUT)
    except FileNotFoundError:
        return {"ok": False, "error": "git is not installed"}
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"git timed out after {GIT_TIMEOUT}s"}
    if r.returncode != 0:
        return {"ok": False, "error": (r.stderr or "git error").strip()[:500]}
    return {"ok": True, "result": (r.stdout or "").strip()[:MAX_READ_BYTES]}


def _fs_list(config, args):
    ok, detail = perm.check_fs_read(config, args.get("path", ""))
    if not ok:
        return {"ok": False, "error": detail}
    if not os.path.isdir(detail):
        return {"ok": False, "error": f"not a directory: {detail}"}
    entries = []
    for name in sorted(os.listdir(detail))[:500]:
        full = os.path.join(detail, name)
        entries.append(f"{'d' if os.path.isdir(full) else 'f'}  {name}")
    return {"ok": True, "result": "\n".join(entries)}


def _fs_read(config, args):
    ok, detail = perm.check_fs_read(config, args.get("path", ""))
    if not ok:
        return {"ok": False, "error": detail}
    if not os.path.isfile(detail):
        return {"ok": False, "error": f"not a file: {detail}"}
    try:
        with open(detail, "r", encoding="utf-8", errors="replace") as f:
            data = f.read(MAX_READ_BYTES + 1)
    except OSError as e:
        return {"ok": False, "error": f"read failed: {e}"}
    truncated = len(data) > MAX_READ_BYTES
    return {"ok": True, "result": data[:MAX_READ_BYTES],
            "truncated": truncated}


def _git(config, args, git_args_fn):
    ok, detail = perm.check_git(config, args.get("repo", ""))
    if not ok:
        return {"ok": False, "error": detail}
    return _run_git(detail, git_args_fn(args))


DISPATCH = {
    "fs_list": _fs_list,
    "fs_read": _fs_read,
    "git_status": lambda c, a: _git(c, a, lambda a: ["status", "--short", "--branch"]),
    "git_diff": lambda c, a: _git(c, a, lambda a: ["diff"]),
    "git_log": lambda c, a: _git(
        c, a, lambda a: ["log", "--oneline", "-n", str(min(max(int(a.get("n", 10)), 1), 50))]),
}


def dispatch(call: dict, config: dict | None = None) -> dict:
    """Validate and execute one tool call. Never raises for a bad/denied call."""
    if config is None:
        config = perm.load_config()
    name = (call or {}).get("tool")
    args = (call or {}).get("arguments", {}) or {}
    fn = DISPATCH.get(name)
    if fn is None:
        return {"ok": False, "error": f"unknown tool {name!r}; available: {list(DISPATCH)}"}
    try:
        return fn(config, args)
    except Exception as e:                                       # noqa: BLE001
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}
