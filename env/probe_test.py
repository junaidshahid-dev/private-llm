"""probe_test.py — environment awareness returns real facts, never raises, redacts on request.

    python env/probe_test.py
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

from env.probe import probe, facts_block, os_info, which_tools, git_info       # noqa: E402
from privacy.redact import Profile                                            # noqa: E402

fails = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))
    if not ok:
        fails.append(name)


def main() -> int:
    print("=" * 74)
    print("ENVIRONMENT AWARENESS TEST — real facts, no crashes, privacy-aware")
    print("=" * 74)

    o = os_info()
    check("os_info reports a system", bool(o["system"]))
    check("os_info reports python version", o["python"][0].isdigit())

    tools = which_tools()
    check("python is detected (it is running us)",
          "python" in tools or "python3" in tools, str(sorted(tools)))
    check("detected tools carry a path", all(t.get("path") for t in tools.values()))

    g = git_info(HERE)
    check("git_info detects the repo here", g.get("available") and g.get("is_repo"),
          str({k: g.get(k) for k in ("is_repo", "branch")}))

    facts = probe()
    for key in ("os", "cwd", "tools", "git", "docker", "adb_devices"):
        check(f"probe() includes '{key}'", key in facts)
    check("probe never invents a device", isinstance(facts["adb_devices"], list))

    block = facts_block(facts)
    check("facts_block tells the model to reason from facts, not guess",
          "do not guess" in block and "working directory" in block)
    check("facts_block lists the real cwd", facts["cwd"] in block)

    me = Profile(home_paths={os.path.expanduser("~")})
    red = facts_block(facts, redact_profile=me)
    home = os.path.expanduser("~")
    if home in block and home != "~":
        check("privacy redaction hides the home path in the facts block", home not in red)
    else:
        check("privacy redaction path (home not present; trivially ok)", True)

    print("\n" + "=" * 74)
    if fails:
        print(f"FAILED {len(fails)}: {', '.join(fails)}")
        return 1
    print("ALL ENVIRONMENT TESTS PASS — the assistant can reason from real surroundings.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
