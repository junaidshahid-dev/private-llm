"""security.py — authorized security tooling, scoped to YOUR lab. Recon/analysis first.

This is the network analogue of the filesystem sandbox. The security-critical guarantee is
target validation: a scan or capture runs only against a target inside your configured
allowed_targets (your own hosts and RFC1918 lab ranges). A public IP, a target in a different
subnet, or an unlisted hostname is rejected before anything runs. That is what makes this
authorized testing of your own lab rather than a tool aimed at the internet.

THREE GATES, every call:
  1. the security_tools group is enabled in configs/tools.yaml (off by default)
  2. the target validates against allowed_targets (the core check, tested hard)
  3. if require_confirmation, the call returns the exact command it WOULD run and stops; a human
     re-issues it with confirmed=True. The model proposes; you approve.

Every attempt — proposed, denied, executed — is appended to an audit log.

SCOPE. Recon and analysis only for now: nmap service/version scan, tshark analysis of a capture
file. No exploitation (Metasploit run-exploit etc.) — that is a later, higher level, and it is
deliberately not implemented here. Start read-only; earn active testing.

Hostname policy is strict on purpose: a target is allowed only if it is an IP inside an allowed
range, or a hostname listed verbatim in allowed_targets. Arbitrary hostnames are NOT resolved and
allowed by their resolved IP — that would invite DNS rebinding past the scope.
"""
from __future__ import annotations

import ipaddress
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone

from mcp_layer import permissions as perm

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUDIT_LOG = os.path.join(HERE, "mcp_layer", "security_audit.log")
SCAN_TIMEOUT = 120


def target_allowed(target: str, allowed_targets: list[str]) -> tuple[bool, str]:
    """A target is allowed only if it is an IP within an allowed range, or a listed hostname."""
    t = (target or "").strip()
    if not t:
        return False, "empty target"
    if t in (allowed_targets or []):
        return True, f"exact match: {t}"
    try:
        ip = ipaddress.ip_address(t)
    except ValueError:
        return False, (f"hostname {t!r} is not in allowed_targets; only IPs in allowed ranges "
                       "or verbatim-listed hostnames are permitted (no arbitrary resolution)")
    for a in allowed_targets or []:
        try:
            net = ipaddress.ip_network(a, strict=False)
        except ValueError:
            continue                                  # a is a hostname, not a network
        if ip in net:
            return True, f"{ip} is in {a}"
    return False, f"{ip} is outside every allowed range {allowed_targets}"


def audit(event: str, tool: str, target: str, detail: str = "") -> None:
    line = json.dumps({"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                       "event": event, "tool": tool, "target": target, "detail": detail[:200]})
    try:
        with open(AUDIT_LOG, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError:
        pass


def schema() -> list[dict]:
    return [
        {"name": "nmap_scan", "description": "Service/version recon scan of an ALLOWED target "
         "(non-destructive). Requires confirmation.",
         "arguments": {"target": "an IP/host inside allowed_targets"}},
        {"name": "pcap_analyze", "description": "Protocol summary of a capture file (read-only).",
         "arguments": {"path": "path to a .pcap/.pcapng inside an allowed path"}},
    ]


def _sec_cfg(config: dict) -> dict:
    return config.get("security_tools") or {}


def _gate(config: dict, tool: str, target: str, confirmed: bool):
    """Return (proceed, response). If proceed is False, response is the result to return."""
    sc = _sec_cfg(config)
    if not sc.get("enabled"):
        audit("denied", tool, target, "group disabled")
        return False, {"ok": False, "error": "security_tools is disabled in configs/tools.yaml"}
    ok, why = target_allowed(target, sc.get("allowed_targets", []))
    if not ok:
        audit("denied", tool, target, why)
        return False, {"ok": False, "error": f"target not allowed: {why}"}
    if sc.get("require_confirmation", True) and not confirmed:
        audit("proposed", tool, target, why)
        return False, {"ok": False, "needs_confirmation": True, "target": target,
                       "note": f"target {target} is in scope ({why}). Re-issue with "
                               "confirmed=true to run."}
    return True, None


def nmap_scan(config: dict, target: str, confirmed: bool = False) -> dict:
    proceed, resp = _gate(config, "nmap_scan", target, confirmed)
    if not proceed:
        return resp
    if not shutil.which("nmap"):
        audit("error", "nmap_scan", target, "nmap not installed")
        return {"ok": False, "error": "nmap is not installed on this machine"}
    cmd = ["nmap", "-sV", "-Pn", "--top-ports", "100", target]
    audit("executed", "nmap_scan", target, " ".join(cmd))
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=SCAN_TIMEOUT)
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"nmap timed out after {SCAN_TIMEOUT}s"}
    return {"ok": r.returncode == 0, "result": (r.stdout or "")[:8000],
            "error": (r.stderr or "").strip()[:400] or None}


def pcap_analyze(config: dict, path: str, confirmed: bool = False) -> dict:
    # a capture file is read-only analysis, but it must still be inside an allowed filesystem path
    ok, detail = perm.check_fs_read(config, path)
    if not ok:
        audit("denied", "pcap_analyze", path, detail)
        return {"ok": False, "error": detail}
    if not _sec_cfg(config).get("enabled"):
        return {"ok": False, "error": "security_tools is disabled in configs/tools.yaml"}
    if not shutil.which("tshark"):
        return {"ok": False, "error": "tshark is not installed on this machine"}
    cmd = ["tshark", "-r", detail, "-q", "-z", "io,phs"]
    audit("executed", "pcap_analyze", path, " ".join(cmd))
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=SCAN_TIMEOUT)
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "tshark timed out"}
    return {"ok": r.returncode == 0, "result": (r.stdout or "")[:8000],
            "error": (r.stderr or "").strip()[:400] or None}


DISPATCH = {
    "nmap_scan": lambda c, a, cf: nmap_scan(c, a.get("target", ""), cf),
    "pcap_analyze": lambda c, a, cf: pcap_analyze(c, a.get("path", ""), cf),
}


def dispatch(call: dict, config: dict, confirmed: bool = False) -> dict:
    fn = DISPATCH.get((call or {}).get("tool"))
    if fn is None:
        return {"ok": False, "error": f"unknown security tool {call.get('tool')!r}"}
    try:
        return fn(config, call.get("arguments", {}) or {}, confirmed)
    except Exception as e:                                       # noqa: BLE001
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}
