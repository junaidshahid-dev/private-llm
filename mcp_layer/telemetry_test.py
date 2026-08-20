"""telemetry_test.py — the audit chain hashes tool output and redacts secrets.

    python mcp_layer/telemetry_test.py
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

from mcp_layer.telemetry import Telemetry, output_hash                       # noqa: E402

fails = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))
    if not ok:
        fails.append(name)


def main() -> int:
    print("=" * 70)
    print("TELEMETRY — full auditable chain, output hashing, secret redaction")
    print("=" * 70)

    print("\n1. output_hash is stable and discriminating")
    a = {"ok": True, "result": {"ports": [80, 443]}}
    check("same input -> same hash", output_hash(a) == output_hash(dict(a)))
    check("different input -> different hash", output_hash(a) != output_hash({"ok": False}))
    check("hash is a sha256 hex", len(output_hash(a)) == 64)

    print("\n2. tool_result records a hash of the real output + a redacted preview")
    t = Telemetry("s1")
    out = {"ok": True, "result": "AWS key AKIAIOSFODNN7EXAMPLE found in config"}
    rec = t.tool_result("http_get", "lab.local", True, out)
    check("output_sha256 matches output_hash(output)", rec["output_sha256"] == output_hash(out))
    check("the preview does NOT contain the raw secret",
          "AKIAIOSFODNN7EXAMPLE" not in rec["output_preview"], rec["output_preview"])

    print("\n3. free-text fields are redacted")
    r = t.interpretation("the credential AKIAIOSFODNN7EXAMPLE is exposed")
    check("interpretation redacts the secret", "AKIAIOSFODNN7EXAMPLE" not in r["text"])

    print("\n4. the full chain is recorded in order")
    t2 = Telemetry("s2")
    t2.instruction("assess the lab web app")
    t2.plan("recon then investigate /config")
    t2.proposal("nmap_scan", {"target": "lab.local"}, "service discovery")
    t2.authorization("nmap_scan", "lab.local", True, "in session scope")
    t2.tool_result("nmap_scan", "lab.local", True, {"ports": [80]})
    t2.interpretation("80/tcp open; investigate the web app next")
    t2.verdict("PASS", [])
    t2.report("1 hypothesis, 0 confirmed")
    kinds = t2.kinds()
    for k in ("instruction", "plan", "proposal", "authorization", "tool_result", "interpretation",
              "verification", "report"):
        check(f"chain has a {k} event", k in kinds)
    check("events are in chronological order", kinds[0] == "instruction" and kinds[-1] == "report")

    print("\n5. a failing sink never breaks the run")
    def bad_sink(_rec):
        raise RuntimeError("disk full")
    ts = Telemetry("s3", sink=bad_sink)
    ts.plan("x")                                       # must not raise
    check("record survives a raising sink", ts.kinds() == ["plan"])

    print("\n" + "=" * 70)
    if fails:
        print(f"FAILED {len(fails)}: {', '.join(fails)}")
        return 1
    print("ALL TELEMETRY TESTS PASS — every step recorded, output hashed, secrets redacted.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
