"""graph_test.py — the assessment state graph: ingest real tool output, steer the next observation.

    python assessment/graph_test.py

Uses the ACTUAL lab outputs (nmap Apache 2.4.25, ffuf DVWA paths). Proves the graph captures state,
keeps a banner as INFERRED (not confirmed), and — the point — flags a re-scan as redundant while
pointing at the discovered HTTP service. That is the fix for the measured masscan tool-selection
weakness.
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

from assessment.graph import AssessmentState, parse_nmap, parse_ffuf         # noqa: E402

fails = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))
    if not ok:
        fails.append(name)


NMAP = """Nmap scan report for web-target (172.28.0.10)
Host is up (0.000015s latency).
PORT   STATE SERVICE VERSION
80/tcp open  http    Apache httpd 2.4.25 ((Debian))
"""
FFUF = "login.php\nsetup.php\nconfig\nphpinfo.php\nindex.php\nrobots.txt\nserver-status"


def main() -> int:
    print("=" * 72)
    print("ASSESSMENT STATE GRAPH TEST — ingest real recon, steer the next observation")
    print("=" * 72)

    print("\n1. parsers")
    ports = parse_nmap(NMAP)
    check("nmap: port 80 http Apache parsed", ports and ports[0]["port"] == 80
          and "Apache" in ports[0]["version"], str(ports))
    paths = parse_ffuf(FFUF)
    check("ffuf: paths parsed with leading slash", "/login.php" in paths and "/config" in paths)

    print("\n2. ingest nmap -> ports + INFERRED hypothesis (banner is not proof)")
    st = AssessmentState()
    st.update_from_tool("nmap_scan", {"target": "web-target"}, {"ok": True, "result": NMAP})
    check("host + port 80 recorded", st.hosts["web-target"]["ports"][80]["service"] == "http")
    check("Apache banner kept as INFERRED, not confirmed",
          any(h["status"] == "inferred" and "Apache" in h["text"] for h in st.hypotheses))
    check("no CONFIRMED vuln invented from a banner",
          not any(h["status"] == "confirmed" for h in st.hypotheses))

    print("\n3. next observation after a port scan — DON'T re-scan, investigate the web service")
    obs = st.next_observations()
    check("a second port scan is flagged REDUNDANT",
          any(o["redundant"] and "port scan" in o["action"] for o in obs))
    check("web content discovery is suggested (actionable)",
          any(not o["redundant"] and "web content" in o["action"] for o in obs))
    check("actionable comes before redundant", obs and obs[0]["redundant"] is False)

    print("\n4. ingest ffuf -> endpoints, then high-value ones are prioritised")
    st.update_from_tool("ffuf_discover", {"target": "web-target"}, {"ok": True, "result": FFUF})
    check("endpoints recorded", "/config" in st.hosts["web-target"]["endpoints"])
    obs2 = st.next_observations()
    hv = [o["action"] for o in obs2 if not o["redundant"]]
    check("high-value endpoints (config/phpinfo/setup) proposed for inspection",
          any("config" in a or "phpinfo" in a or "setup.php" in a for a in hv), str(hv)[:120])
    check("web content discovery no longer suggested (already mapped)",
          not any("web content" in o["action"] for o in obs2))

    print("\n5. ingest http_get -> endpoint status + technology")
    st.update_from_tool("http_get", {"url": "http://web-target/login.php"},
                        {"ok": True, "result": {"status": 200, "final_url": "http://web-target/login.php",
                                                "headers": {"Server": "Apache/2.4.25"}}})
    check("endpoint status recorded", st.hosts["web-target"]["endpoints"]["/login.php"]["status"] == 200)
    check("technology recorded from Server header",
          "Apache/2.4.25" in st.hosts["web-target"]["technologies"])

    print("\n6. render")
    r = st.render()
    check("render shows host, ports, endpoints, next observations",
          "web-target" in r and "80/" in r and "next observations" in r)

    print("\n" + "=" * 72)
    if fails:
        print(f"FAILED {len(fails)}: {', '.join(fails)}")
        return 1
    print("ALL ASSESSMENT-GRAPH TESTS PASS — stateful, evidence-graded, steers off redundant scans.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
