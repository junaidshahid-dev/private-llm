"""ir_test.py — IOC extraction, log timeline, and tamper/anomaly detection.

    python analysis/ir_test.py
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

from analysis.ir import extract_iocs, log_timeline, log_anomalies, analyze_log   # noqa: E402

fails = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))
    if not ok:
        fails.append(name)


def main() -> int:
    print("=" * 70)
    print("IR — IOC extraction, log timeline, tamper/anomaly signs")
    print("=" * 70)

    print("\n1. IOC extraction (low false positive)")
    text = ("attacker 203.0.113.9 fetched http://evil.example/x, dropped a file "
            "sha256 " + "a" * 64 + " and CVE-2021-44228 was used; contact bad@evil.com")
    iocs = extract_iocs(text)
    check("extracts the IPv4", "203.0.113.9" in iocs.get("ipv4", []))
    check("extracts the URL", any("evil.example" in u for u in iocs.get("url", [])))
    check("extracts the sha256", "a" * 64 in iocs.get("sha256", []))
    check("extracts the CVE", "CVE-2021-44228" in iocs.get("cve", []))
    check("extracts the email", "bad@evil.com" in iocs.get("email", []))
    check("a filename like config.txt is NOT reported as a domain",
          "config.txt" not in extract_iocs("edited config.txt").get("domain", []))

    print("\n2. log timeline (ISO + syslog, sorted)")
    log = ("2026-08-20T10:00:05 login ok\n2026-08-20T09:59:00 service start\n"
           "not a timestamped line\nAug 20 09:58:00 sshd started")
    tl = log_timeline(log)
    check("parses timestamped lines only", len(tl) == 3, str(len(tl)))
    check("ISO events are sorted ascending",
          tl[0]["line"].endswith("service start") and "10:00:05" in tl[1]["ts"])

    print("\n3. tampering + auth-failure anomalies")
    tamper = log_anomalies("root: history -c\nlogs cleared by intruder")
    check("flags shell-history clearing", any("history" in a["detail"] for a in tamper))
    check("flags logs cleared", any("cleared" in a["detail"] for a in tamper))
    brute = log_anomalies("\n".join(["sshd: Failed password for root"] * 6))
    check("flags an auth-failure burst", any(a["sign"] == "auth_failures" for a in brute))
    check("a clean log has no anomalies", log_anomalies("2026-08-20T10:00:00 normal op") == [])

    print("\n4. analyze_log composes everything")
    a = analyze_log(text + "\n2026-08-20T10:00:00 event")
    check("analyze_log returns events/iocs/anomalies",
          "events" in a and "iocs" in a and "anomalies" in a)

    print("\n" + "=" * 70)
    if fails:
        print(f"FAILED {len(fails)}: {', '.join(fails)}")
        return 1
    print("ALL IR TESTS PASS — IOCs extracted, timeline built, tampering/brute-force flagged.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
