"""probe.py — Environment awareness: give the assistant FACTS about where it is running, so it
reasons from reality instead of guessing (it guessed '/path/to/...' and '~/Desktop/LLM' live).

    from env.probe import probe, facts_block
    facts = probe()                   # a dict of read-only facts
    print(facts_block(facts))         # compact text to put in the model's prompt

(Named probe, not 'inspect', so it does not shadow the stdlib inspect module.)

Read-only and defensive: it runs `--version`-style probes and reads state, never mutates anything,
and never raises (a missing tool or a failed probe just yields None/absent). What it gathers:
OS/kernel, Python + venv, the CWD and repo root, which security/dev TOOLS are actually installed
and their versions (so the model proposes tools that exist), git repo state, Docker presence, and
ADB devices.

PRIVACY: network/host details here are YOUR data. facts_block(..., redact_profile=...) runs the
Privacy layer so nothing identifying leaks if the block is ever sent to an external service. For the
LOCAL model the raw facts are fine.
"""
from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys

# Tools worth probing: recon/web/traffic/RE/mobile/programming + the version flag to ask each.
TOOLS = {
    "nmap": "--version", "masscan": "--version", "ffuf": "-V", "nikto": "-Version",
    "sqlmap": "--version", "tshark": "--version", "wireshark": "--version",
    "gdb": "--version", "radare2": "-v", "r2": "-v", "objdump": "--version",
    "adb": "--version", "git": "--version", "docker": "--version", "python": "--version",
    "python3": "--version", "bash": "--version", "pwsh": "--version", "curl": "--version",
    "openssl": "version", "ip": "-V", "ifconfig": "--version",
}
_PROBE_TIMEOUT = 4


def _run(args: list[str]) -> str | None:
    try:
        r = subprocess.run(args, capture_output=True, text=True, timeout=_PROBE_TIMEOUT)
    except (OSError, subprocess.SubprocessError):
        return None
    out = ((r.stdout or "") + (r.stderr or "")).strip()
    return out.splitlines()[0].strip() if out else ""


def os_info() -> dict:
    return {"system": platform.system(), "release": platform.release(),
            "machine": platform.machine(), "platform": platform.platform(),
            "python": platform.python_version(), "python_exe": sys.executable,
            "in_venv": sys.prefix != getattr(sys, "base_prefix", sys.prefix)}


def which_tools(names: dict | None = None) -> dict:
    names = names or TOOLS
    found = {}
    for name, flag in names.items():
        path = shutil.which(name)
        if path:
            found[name] = {"path": path, "version": _run([name, flag])}
    return found


def git_info(path: str) -> dict:
    if not shutil.which("git"):
        return {"available": False}
    top = _run(["git", "-C", path, "rev-parse", "--show-toplevel"])
    if not top:
        return {"available": True, "is_repo": False}
    return {"available": True, "is_repo": True, "root": top,
            "branch": _run(["git", "-C", path, "rev-parse", "--abbrev-ref", "HEAD"]),
            "head": _run(["git", "-C", path, "rev-parse", "--short", "HEAD"])}


def docker_info() -> dict:
    if not shutil.which("docker"):
        return {"available": False}
    return {"available": True,
            "running": _run(["docker", "info", "--format", "{{.ServerVersion}}"]) not in (None, "")}


def adb_devices() -> list[str]:
    if not shutil.which("adb"):
        return []
    try:
        r = subprocess.run(["adb", "devices"], capture_output=True, text=True,
                           timeout=_PROBE_TIMEOUT)
        return [ln.split()[0] for ln in (r.stdout or "").splitlines()[1:]
                if ln.strip() and "\t" in ln]
    except (OSError, subprocess.SubprocessError):
        return []


def probe(cwd: str | None = None) -> dict:
    cwd = cwd or os.getcwd()
    return {"os": os_info(), "cwd": cwd, "tools": which_tools(),
            "git": git_info(cwd), "docker": docker_info(), "adb_devices": adb_devices()}


def facts_block(facts: dict, redact_profile=None) -> str:
    o = facts["os"]
    tools = facts["tools"]
    lines = ["ENVIRONMENT FACTS (read-only; reason from these, do not guess):",
             f"- OS: {o['system']} {o['release']} ({o['machine']}); Python {o['python']}"
             f"{' [venv]' if o['in_venv'] else ''}",
             f"- working directory: {facts['cwd']}"]
    g = facts["git"]
    if g.get("is_repo"):
        lines.append(f"- git repo: {g.get('root')} @ {g.get('branch')} ({g.get('head')})")
    if tools:
        lines.append(f"- installed tools available to propose: {', '.join(sorted(tools))}")
        missing = [t for t in ("nmap", "tshark", "adb", "docker", "radare2") if t not in tools]
        if missing:
            lines.append(f"- NOT installed (do not propose): {', '.join(missing)}")
    if facts["docker"].get("available"):
        lines.append(f"- docker present (daemon running: {facts['docker'].get('running')})")
    if facts["adb_devices"]:
        lines.append(f"- ADB devices connected: {len(facts['adb_devices'])}")
    block = "\n".join(lines)
    if redact_profile is not None:
        from privacy.redact import redact
        block = redact(block, redact_profile)[0]
    return block


if __name__ == "__main__":
    print(facts_block(probe()))
