"""permissions.py — the security-critical core of the tool layer. DENY BY DEFAULT.

Every tool call is checked here before anything happens. This is the file to read closely and
distrust: if it is wrong, the model can reach outside its sandbox. It is deliberately small so it
can be audited at a glance.

TWO GUARANTEES

  path_allowed: a filesystem path is permitted only if, AFTER resolving symlinks and '..', it
                lies inside one of the configured allowed roots. Resolution happens first, so
                "~/Desktop/LLM/../../Windows/System32" resolves to a path outside the root and is
                rejected. There is no string-prefix check that a "/../" could fool.

  command_allowed: a shell command is permitted only if its first token is in the explicit
                allow-list. (Terminal is off by default; this is here for when it is enabled.)

Anything not explicitly enabled and in scope returns (False, reason). The caller must treat a
False as a hard stop and surface the reason to the model, never execute.
"""
from __future__ import annotations

import os
from pathlib import Path

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(HERE, "configs", "tools.yaml")


def load_config(path: str = CONFIG_PATH) -> dict:
    import yaml
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _resolve(p: str) -> Path | None:
    try:
        return Path(p).expanduser().resolve()
    except (OSError, RuntimeError, ValueError):
        return None


def path_allowed(path: str, allowed_roots: list[str]) -> tuple[bool, str]:
    """True only if the RESOLVED path is inside a resolved allowed root."""
    target = _resolve(path)
    if target is None:
        return False, f"cannot resolve path: {path!r}"
    for root in allowed_roots or []:
        r = _resolve(root)
        if r is None:
            continue
        if target == r or target.is_relative_to(r):
            return True, str(target)
    return False, f"path {str(target)!r} is outside the allowed roots {allowed_roots}"


def command_allowed(command: str, allowed_commands: list[str]) -> tuple[bool, str]:
    """True only if the command's first token is in the allow-list."""
    parts = (command or "").strip().split()
    if not parts:
        return False, "empty command"
    head = os.path.basename(parts[0])
    if head in (allowed_commands or []):
        return True, head
    return False, f"command {head!r} not in allow-list {allowed_commands}"


def tool_enabled(config: dict, group: str) -> tuple[bool, str]:
    g = config.get(group) or {}
    if not g.get("enabled"):
        return False, f"tool group {group!r} is disabled in configs/tools.yaml"
    return True, ""


def check_fs_read(config: dict, path: str) -> tuple[bool, str]:
    ok, why = tool_enabled(config, "filesystem_read")
    if not ok:
        return ok, why
    return path_allowed(path, (config["filesystem_read"] or {}).get("allowed_paths", []))


def check_git(config: dict, repo: str) -> tuple[bool, str]:
    ok, why = tool_enabled(config, "git_inspect")
    if not ok:
        return ok, why
    return path_allowed(repo, (config["git_inspect"] or {}).get("allowed_repos", []))
