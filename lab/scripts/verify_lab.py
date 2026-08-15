"""verify_lab.py — pre-flight for the controlled lab. Run before any active test.

    python lab/scripts/verify_lab.py

The validation sequence:
    connectivity to the target
      -> confirm the authorized target RESOLVES TO THE LAB (loopback / private / labnet), never a
         public IP  [the hard safety gate]
      -> confirm the MCP config AUTHORIZES it
      -> binary-verify which tools are actually installed
      -> print the readiness ladder per tool

The safety gate is the point: even if something is misconfigured, an active tool must never be
green-lit against a public address. resolves_to_lab() refuses anything that is not loopback,
RFC-1918 private, or inside the labnet subnet. That logic is unit-tested (verify_lab_test.py) and
needs no Docker.
"""
from __future__ import annotations

import ipaddress
import os
import shutil
import socket
import sys

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, HERE)

LAB_SUBNET = ipaddress.ip_network("172.28.0.0/24")
LAB_FILE = os.path.join(HERE, "lab", "authorized_targets.yaml")
DEFAULT_TARGET = "127.0.0.1:8080"
ACTIVE_TOOLS = ("nmap", "ffuf", "masscan")


def _host_port(target: str):
    t = (target or "").strip()
    if "://" in t:
        from urllib.parse import urlparse
        u = urlparse(t)
        return u.hostname, u.port or (443 if u.scheme == "https" else 80)
    if t.count(":") == 1 and "[" not in t:
        h, _, p = t.partition(":")
        return h, int(p) if p.isdigit() else None
    return t, None


def resolve_host(host: str):
    try:
        return socket.gethostbyname(host)
    except (socket.gaierror, OSError):
        return None


def is_lab_ip(ip: str) -> bool:
    try:
        a = ipaddress.ip_address(ip)
    except ValueError:
        return False
    return a.is_loopback or a.is_private or (a in LAB_SUBNET)


def resolves_to_lab(target: str):
    """(ok, ip, reason). The hard safety gate: refuse anything that is not lab/loopback/private."""
    host, _ = _host_port(target)
    if not host:
        return False, None, "no host in target"
    # a bare container hostname (e.g. web-target) that only resolves inside docker is allowed by name
    ip = resolve_host(host)
    if ip is None:
        if host in ("web-target", "lab-web-target", "operator", "lab-operator"):
            return True, None, f"lab container name {host!r} (resolves only inside labnet)"
        return False, None, f"cannot resolve host {host!r}"
    if is_lab_ip(ip):
        return True, ip, f"{host} -> {ip} is loopback/private/labnet"
    return False, ip, f"REFUSED: {host} -> {ip} is a PUBLIC address, not the lab"


def binary_status(tools=ACTIVE_TOOLS) -> dict:
    return {t: shutil.which(t) for t in tools}


def connectivity(host: str, port: int, timeout: float = 3.0) -> bool:
    if not host or not port:
        return False
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def config_authorizes(target: str):
    """Does the LIVE configs/tools.yaml authorize this target through the real MCP gate?"""
    try:
        from mcp_layer import permissions as perm
        from mcp_layer.security import target_authorized
        cfg = perm.load_config()
        sc = cfg.get("security_tools") or {}
        return target_authorized(target, sc.get("authorized_targets", []))
    except Exception as e:                                   # noqa: BLE001
        return False, f"could not evaluate config: {e}"


def main() -> int:
    target = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_TARGET
    host, port = _host_port(target)
    print("=" * 70)
    print(f"LAB VERIFICATION — target {target!r}")
    print("=" * 70)

    ok_safe, ip, reason = resolves_to_lab(target)
    print(f"[safety] resolves to lab: {'OK' if ok_safe else 'REFUSED'} — {reason}")

    conn = connectivity(host, port) if port else None
    print(f"[connectivity] {host}:{port} -> {'reachable' if conn else 'not reachable' if port else 'n/a'}"
          + ("  (is the lab up? `docker compose -f lab/docker-compose.yml up -d`)" if conn is False else ""))

    authz, why = config_authorizes(target)
    print(f"[authorization] MCP config authorizes target: {'yes' if authz else 'NO'} — {why}")
    if not authz:
        print(f"    add the entries from {os.path.relpath(LAB_FILE, HERE)} to "
              "configs/tools.yaml under security_tools.authorized_targets")

    print("[binaries] active tool availability:")
    for tool, path in binary_status().items():
        print(f"    {tool:8} {'installed: ' + path if path else 'NOT installed'}")

    print("\nReadiness ladder (per your rule): implemented -> gated -> locally tested -> "
          "binary verified -> live-tested")
    for tool, path in binary_status().items():
        stage = "binary verified (run it against the lab to reach live-tested)" if path \
            else "locally tested (binary NOT installed here)"
        print(f"    {tool:8} -> {stage}")

    ready = ok_safe and authz and (conn or port is None)
    print("\n" + ("GREEN: safety + authorization satisfied — an approved active test may proceed."
                  if ready else
                  "NOT GREEN: resolve the items above before any active test."))
    return 0 if ready else 1


if __name__ == "__main__":
    sys.exit(main())
