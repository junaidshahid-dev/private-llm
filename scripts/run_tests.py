"""run_tests.py — run every *_test.py (+ the build_secv3 self-test) and report. The regression gate.

    python scripts/run_tests.py

Discovers all test modules in the repo, runs each in a subprocess, and reports PASS/FAIL with a
summary. Exit code is non-zero if anything failed, so it can gate a change ("REGRESSION -> don't
ship"). CPU-only; skips the GPU runners (those are measured on Kaggle, not here).
"""
from __future__ import annotations

import glob
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKIP = (".venv", "site-packages", "__pycache__", "scratch", os.sep + "node_modules")

# test modules (self-verifying, exit 0 on pass) — *_test.py plus the benchmark builders' self-tests
extra = [os.path.join(HERE, "evaluation", "development", "security_v3", "build_secv3.py"),
         os.path.join(HERE, "evaluation", "development", "security_v4", "build_secv4.py"),
         os.path.join(HERE, "evaluation", "development", "security_v5", "build_secv5.py"),
         os.path.join(HERE, "evaluation", "pipeline_bench.py"),
         os.path.join(HERE, "evaluation", "tool_bench.py"),
         os.path.join(HERE, "evaluation", "agent_eval.py"),
         os.path.join(HERE, "web", "benchmark.py"),
         os.path.join(HERE, "memory", "benchmark.py"),
         os.path.join(HERE, "tests", "api", "test_webui.py"),   # local-UI backend + E2E (test_ prefix)
         os.path.join(HERE, "tests", "api", "test_remote.py")]  # remote-GPU backend (test_ prefix)


def discover() -> list[str]:
    found = [p for p in glob.glob(os.path.join(HERE, "**", "*_test.py"), recursive=True)
             if not any(s in p for s in SKIP)]
    return sorted(set(found + extra))


def main() -> int:
    py = sys.executable
    tests = discover()
    print("=" * 74)
    print(f"REGRESSION RUN — {len(tests)} test modules")
    print("=" * 74)
    failed = []
    t0 = time.time()
    for t in tests:
        rel = os.path.relpath(t, HERE)
        r = subprocess.run([py, t], capture_output=True, text=True, cwd=HERE)
        ok = r.returncode == 0
        print(f"  [{'PASS' if ok else 'FAIL'}] {rel}")
        if not ok:
            failed.append((rel, (r.stdout or "")[-600:] + (r.stderr or "")[-300:]))
    print("=" * 74)
    if failed:
        print(f"REGRESSION DETECTED — {len(failed)}/{len(tests)} FAILED in {time.time()-t0:.0f}s:")
        for rel, tail in failed:
            print(f"\n--- {rel} ---\n{tail}")
        return 1
    print(f"ALL {len(tests)} TEST MODULES PASS in {time.time()-t0:.0f}s — no regression.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
