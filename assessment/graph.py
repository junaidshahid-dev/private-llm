"""graph.py — the assessment STATE GRAPH: what we know about a target, updated by every tool result.

The architectural fix for the tool-selection weakness measured in the Moonlight/Qwen runs (proposing
masscan when port discovery was already done). Instead of the model choosing a tool from nothing, it
maintains state:

    host -> ports -> services -> endpoints -> technologies ; tests_performed ; findings ; hypotheses

Every tool result UPDATES the graph (parse_nmap / parse_ffuf / http_get headers). Then
next_observations() answers "what is the highest-value AUTHORIZED next observation?" and explicitly
flags REDUNDANT actions (a second port scan when ports are known). Facts carry provenance (which
tool produced them) and hypotheses carry an evidence STATUS
(observed -> inferred -> hypothesis -> tested -> confirmed/rejected) so a banner like 'Apache/2.4.25'
is never treated as a confirmed vulnerability.

Pure data + parsing, fully unit-tested; no model, no network.
"""
from __future__ import annotations

import re
from urllib.parse import urlparse

STATUSES = ("observed", "inferred", "hypothesis", "tested", "confirmed", "rejected")
HIGH_VALUE = ("config", "setup", "admin", "phpinfo", "status", ".git", "backup", "login",
              ".env", "install", "phpmyadmin", "wp-admin", "actuator")


def parse_nmap(output: str) -> list[dict]:
    ports = []
    for m in re.finditer(r"^(\d+)/(tcp|udp)\s+(\w+)\s+(\S+)(?:\s+(.*))?$", output or "", re.M):
        ports.append({"port": int(m.group(1)), "proto": m.group(2), "state": m.group(3),
                      "service": m.group(4), "version": (m.group(5) or "").strip()})
    return ports


def parse_ffuf(output: str) -> list[str]:
    """ffuf -s prints one discovered path per line."""
    out = []
    for ln in (output or "").splitlines():
        ln = ln.strip()
        if ln and re.fullmatch(r"[\w./~%-]+", ln) and not ln.startswith(":"):
            out.append("/" + ln.lstrip("/"))
    return out


def _host(target: str) -> str:
    t = (target or "").strip()
    if "://" in t:
        return urlparse(t).hostname or t
    return t.split("/")[0].split(":")[0]


class AssessmentState:
    def __init__(self):
        self.hosts: dict[str, dict] = {}
        self.tests: list[dict] = []
        self.findings: list[dict] = []
        self.hypotheses: list[dict] = []

    def host(self, h: str) -> dict:
        return self.hosts.setdefault(h, {"ports": {}, "endpoints": {}, "technologies": []})

    def note_test(self, tool: str, target: str, summary: str = "") -> None:
        self.tests.append({"tool": tool, "target": target, "summary": summary})

    def add_hypothesis(self, text: str, status: str = "hypothesis", evidence: str = "") -> None:
        self.hypotheses.append({"text": text, "status": status if status in STATUSES else "hypothesis",
                                "evidence": evidence})

    # ---- ingest a tool result ------------------------------------------------
    def update_from_tool(self, tool: str, arguments: dict, result: dict) -> None:
        args = arguments or {}
        target = args.get("target") or args.get("url") or ""
        host = _host(target)
        res = (result or {}).get("result", result)

        if tool in ("nmap_scan", "masscan_scan"):
            self.note_test(tool, host, "port/service scan")
            for p in parse_nmap(res if isinstance(res, str) else ""):
                rec = self.host(host)["ports"].setdefault(p["port"], {})
                rec.update({"service": p["service"], "version": p["version"], "state": p["state"],
                            "via": tool})
                if p["version"]:
                    self.add_hypothesis(f"{host}:{p['port']} may run {p['version']}",
                                        status="inferred", evidence=f"{tool} banner")
        elif tool == "ffuf_discover":
            self.note_test(tool, host, "web content discovery")
            for path in parse_ffuf(res if isinstance(res, str) else ""):
                self.host(host)["endpoints"].setdefault(path, {"via": "ffuf_discover"})
        elif tool in ("http_get", "web_fetch", "web_extract"):
            self.note_test(tool, host, "http fetch")
            if isinstance(res, dict):
                path = urlparse(res.get("final_url", target)).path or "/"
                ep = self.host(host)["endpoints"].setdefault(path, {})
                ep.update({"status": res.get("status"), "via": tool})
                server = (res.get("headers") or {}).get("Server") or \
                    (res.get("headers") or {}).get("server")
                if server and server not in self.host(host)["technologies"]:
                    self.host(host)["technologies"].append(server)

    def port_scanned(self, host: str) -> bool:
        return any(t["tool"] in ("nmap_scan", "masscan_scan") and t["target"] == host
                   for t in self.tests)

    # ---- the intelligence: highest-value authorized next observation ---------
    def next_observations(self) -> list[dict]:
        obs = []
        for h, hd in self.hosts.items():
            web_ports = [p for p, pd in hd["ports"].items()
                         if "http" in (pd.get("service") or "") or p in (80, 443, 8080, 8443)]
            if self.port_scanned(h):
                obs.append({"action": f"another port scan of {h}", "redundant": True,
                            "why": "port discovery already done — do NOT re-scan; act on what was found"})
            if web_ports and not hd["endpoints"]:
                obs.append({"action": f"discover web content on {h} (ffuf_discover)",
                            "redundant": False,
                            "why": f"HTTP service on {web_ports} but endpoints not yet mapped"})
            for path, ed in hd["endpoints"].items():
                if any(k in path.lower() for k in HIGH_VALUE) and "status" not in ed:
                    obs.append({"action": f"fetch http://{h}{path} (http_get)", "redundant": False,
                                "why": "high-value endpoint discovered but not yet inspected"})
        obs.sort(key=lambda o: o["redundant"])          # actionable first, redundant last
        return obs

    def render(self) -> str:
        L = ["ASSESSMENT STATE"]
        for h, hd in self.hosts.items():
            L.append(f"{h}")
            for p in sorted(hd["ports"]):
                pd = hd["ports"][p]
                L.append(f"  {p}/{pd.get('state','?')}  {pd.get('service','')} {pd.get('version','')}".rstrip())
            for path in sorted(hd["endpoints"]):
                ed = hd["endpoints"][path]
                L.append(f"    {path}  {ed.get('status','')}".rstrip())
            if hd["technologies"]:
                L.append(f"    tech: {', '.join(hd['technologies'])}")
        if self.hypotheses:
            L.append("hypotheses:")
            for hy in self.hypotheses:
                L.append(f"  [{hy['status']}] {hy['text']}")
        nxt = self.next_observations()
        if nxt:
            L.append("next observations (highest value first):")
            for o in nxt:
                tag = "SKIP(redundant)" if o["redundant"] else "DO"
                L.append(f"  [{tag}] {o['action']} — {o['why']}")
        return "\n".join(L)
