"""verify_lab_test.py — the lab safety gate refuses anything that isn't the lab.

    python lab/scripts/verify_lab_test.py

The one property that MUST hold: an active tool is never green-lit against a public address. Tested
without Docker (pure resolution/classification logic).
"""
from __future__ import annotations

import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except (AttributeError, ValueError):
    pass
HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(HERE, "lab", "scripts"))

from verify_lab import is_lab_ip, resolves_to_lab, _host_port, binary_status   # noqa: E402

fails = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))
    if not ok:
        fails.append(name)


def main() -> int:
    print("=" * 70)
    print("LAB SAFETY GATE TEST — only the lab is ever green-lit")
    print("=" * 70)

    print("\n1. is_lab_ip — loopback/private/labnet allowed, public refused")
    for ip in ("127.0.0.1", "172.28.0.10", "10.0.0.5", "192.168.1.9"):
        check(f"{ip} counts as lab/private", is_lab_ip(ip))
    for ip in ("8.8.8.8", "1.1.1.1", "93.184.216.34"):
        check(f"{ip} is NOT lab (public)", not is_lab_ip(ip))
    check("garbage is not lab", not is_lab_ip("not-an-ip"))

    print("\n2. resolves_to_lab — the hard gate")
    ok, ip, _ = resolves_to_lab("127.0.0.1:8080")
    check("loopback target is allowed", ok and ip == "127.0.0.1")
    ok2, _, reason = resolves_to_lab("8.8.8.8")
    check("PUBLIC ip target is REFUSED", not ok2 and "PUBLIC" in reason.upper(), reason)
    ok3, _, reason3 = resolves_to_lab("http://scanme.nmap.org/")
    check("a public HOSTNAME target is refused (or unresolved)", not ok3, reason3)
    ok4, _, r4 = resolves_to_lab("web-target")
    check("lab container name allowed by name", ok4, r4)

    print("\n3. _host_port parsing")
    check("host:port", _host_port("127.0.0.1:8080") == ("127.0.0.1", 8080))
    check("url", _host_port("http://web-target/")[0] == "web-target")
    check("bare host", _host_port("10.0.0.1") == ("10.0.0.1", None))

    print("\n4. binary_status never raises")
    bs = binary_status()
    check("returns nmap/ffuf/masscan keys", {"nmap", "ffuf", "masscan"} <= set(bs))

    print("\n" + "=" * 70)
    if fails:
        print(f"FAILED {len(fails)}: {', '.join(fails)}")
        return 1
    print("ALL LAB SAFETY TESTS PASS — public targets are refused; only the lab is allowed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
