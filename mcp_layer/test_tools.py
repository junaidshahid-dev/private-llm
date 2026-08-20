"""test_tools.py — prove the tool layer works AND that the sandbox holds.

    python mcp_layer/test_tools.py

The tests that matter most are the DENIALS: a permission layer that allows the good calls but
also allows an escape is worse than none. So this asserts, hard, that path traversal, absolute
paths outside the roots, and disabled tools are all rejected — and only then that the allowed
calls succeed.
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from mcp_layer import permissions as perm       # noqa: E402
from mcp_layer.tools import dispatch, schema     # noqa: E402

fails = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))
    if not ok:
        fails.append(name)


def main() -> int:
    print("=" * 74)
    print("MCP TOOL LAYER — permission + tool tests")
    print("=" * 74)

    # A config scoped to THIS repo only, so escapes are well-defined.
    cfg = {
        "filesystem_read": {"enabled": True, "allowed_paths": [HERE]},
        "git_inspect": {"enabled": True, "allowed_repos": [HERE]},
        "filesystem_write": {"enabled": False},
        "terminal": {"enabled": False, "allowed_commands": ["python", "git"]},
    }

    print("\n1. DENIALS (the ones that matter)")
    # path traversal out of the root
    r = dispatch({"tool": "fs_read", "arguments": {"path": os.path.join(HERE, "..", "..", "..",
                 "Windows", "System32", "drivers", "etc", "hosts")}}, cfg)
    check("traversal '../../..' is blocked", not r["ok"], r.get("error", "")[:50])
    # absolute path outside the root
    out = "C:/Windows/System32/config" if os.name == "nt" else "/etc/passwd"
    r = dispatch({"tool": "fs_read", "arguments": {"path": out}}, cfg)
    check("absolute path outside root is blocked", not r["ok"])
    # home directory outside root
    r = dispatch({"tool": "fs_read", "arguments": {"path": "~/.ssh/id_rsa"}}, cfg)
    check("~/.ssh is blocked", not r["ok"])
    # disabled tool group
    cfg_off = {**cfg, "filesystem_read": {"enabled": False, "allowed_paths": [HERE]}}
    r = dispatch({"tool": "fs_read", "arguments": {"path": os.path.join(HERE, "README.md")}},
                 cfg_off)
    check("disabled tool is blocked even for an in-scope path", not r["ok"])
    # unknown tool
    r = dispatch({"tool": "rm_rf", "arguments": {"path": "/"}}, cfg)
    check("unknown tool is rejected", not r["ok"])
    # write/terminal tools are not implemented at all
    check("no write/exec tools exist in the dispatch table",
          all(t not in schema.__self__.__dict__ if False else
              t["name"] not in ("fs_write", "terminal", "shell", "python_exec")
              for t in schema()))

    print("\n2. ALLOWED read-only calls succeed")
    r = dispatch({"tool": "fs_list", "arguments": {"path": HERE}}, cfg)
    check("fs_list on the repo root works", r["ok"] and "README.md" in r.get("result", ""))
    r = dispatch({"tool": "fs_read", "arguments": {"path": os.path.join(HERE, "MODEL_SPEC.lock.json")}},
                 cfg)
    check("fs_read an allowed file works", r["ok"] and "moonlight" in r.get("result", "").lower())
    r = dispatch({"tool": "git_status", "arguments": {"repo": HERE}}, cfg)
    check("git_status works", r["ok"], r.get("result", r.get("error", ""))[:40])
    r = dispatch({"tool": "git_log", "arguments": {"repo": HERE, "n": 3}}, cfg)
    check("git_log returns commits", r["ok"] and r.get("result", "").count("\n") >= 1)

    print("\n2b. source_scan — read-only static analysis, gated + rich schema")
    import tempfile
    td = tempfile.mkdtemp()
    vp = os.path.join(td, "vuln.py")
    open(vp, "w", encoding="utf-8").write(
        'import os\ncmd = request.args.get("c")\nos.system(cmd)\nAPI_KEY = "AKIAIOSFODNN7EXAMPLE"\n')
    cfg_src = {**cfg, "filesystem_read": {"enabled": True, "allowed_paths": [td]}}
    r = dispatch({"tool": "source_scan", "arguments": {"path": vp}}, cfg_src)
    check("source_scan runs and finds candidates", r["ok"] and len(r.get("findings", [])) >= 2,
          str(len(r.get("findings", []))))
    check("finds the command_injection taint + a hardcoded secret",
          any(f["vuln_class"] == "command_injection" for f in r["findings"])
          and any(f["vuln_class"] == "hardcoded_secret" for f in r["findings"]))
    check("static findings are HYPOTHESES, never CONFIRMED",
          all(f["status"] != "CONFIRMED" for f in r["findings"]))
    check("source_scan is path-confined like fs_read",
          not dispatch({"tool": "source_scan", "arguments": {"path": os.path.join(HERE, "README.md")}},
                       cfg_src)["ok"])
    sdef = next(t for t in schema() if t["name"] == "source_scan")
    check("source_scan declares the rich schema (read_only / verification_method)",
          sdef.get("read_only") is True and sdef.get("side_effects") == "none"
          and sdef.get("verification_method"))

    print("\n3. permission unit checks")
    ok, _ = perm.path_allowed(os.path.join(HERE, "rag"), [HERE])
    check("path_allowed: in-scope subdir allowed", ok)
    ok, _ = perm.path_allowed(os.path.join(HERE, "..", "secrets"), [HERE])
    check("path_allowed: parent escape rejected", not ok)
    ok, _ = perm.command_allowed("rm -rf /", ["python", "git"])
    check("command_allowed: rm rejected", not ok)
    ok, _ = perm.command_allowed("git status", ["python", "git"])
    check("command_allowed: git allowed", ok)

    print("\n" + "=" * 74)
    print(f"FAILED: {fails}" if fails else "ALL TOOL-LAYER TESTS PASSED — sandbox holds.")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
