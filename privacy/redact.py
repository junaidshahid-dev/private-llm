"""redact.py — the Privacy & Network Security Layer: strip the OPERATOR'S OWN sensitive data before
anything is sent to an external service or written to a shared log.

    from privacy.redact import redact, local_profile
    safe, findings = redact(text, profile=local_profile())   # before an external API / a log line

SCOPE, deliberately narrow and defensive:
  * This protects YOUR OWN identifying data (public/private IPs, MACs, hostnames, home paths,
    username, device ids) and ANY secret (keys, tokens, passwords, private keys) from leaving the
    machine unnecessarily. That is the "protect your own infrastructure" goal.
  * It is NOT identity concealment for attacking someone else's system, and NOT attribution
    evasion. Target identifiers (an authorized target's IP/host that are NOT in your own profile)
    pass through untouched, so security reasoning still works.

WHERE IT SITS (per the operator's diagram): between the tools/model and an EXTERNAL boundary —

    your data -> redact() -> external API / web service / shared log

It is NOT applied to the internal tool->model flow: the model needs the real data to reason. Redact
only at the edge where information would leave your control.

Two independent layers, each toggleable:
  redact_secrets   ALWAYS-strip credentials/keys/tokens — these are never safe to send or log.
  redact_identity  strip the operator's own identifiers, matched from a profile (auto-detected from
                   this machine + whatever you add in configs/privacy.yaml).

Pure stdlib, CPU, offline. redact() never raises on bad input; it returns the text it could clean
plus a findings list (category + placeholder + count — never the secret value itself).
"""
from __future__ import annotations

import getpass
import os
import re
import socket
import uuid
from dataclasses import dataclass, field

PLACE = "[REDACTED:{}]"


# ---- always-strip secrets ---------------------------------------------------
# High-confidence secret shapes. Context patterns (key/secret/password = VALUE) capture the VALUE
# group only, so the label stays and the value is masked. Ordered longest/most-specific first.
_SECRETS = [
    ("private_key", re.compile(
        r"-----BEGIN (?:[A-Z0-9 ]+ )?PRIVATE KEY-----.*?-----END (?:[A-Z0-9 ]+ )?PRIVATE KEY-----",
        re.S)),
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_\-]{6,}\.eyJ[A-Za-z0-9_\-]{6,}\.[A-Za-z0-9_\-]{6,}\b")),
    ("aws_key", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("github_token", re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{36,}\b")),
    ("github_pat", re.compile(r"\bgithub_pat_[A-Za-z0-9_]{22,}\b")),
    ("slack_token", re.compile(r"\bxox[baprs]-[A-Za-z0-9\-]{10,}\b")),
    ("bearer", re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._\-]{12,}")),
    ("url_password", re.compile(r"(?i)\b([a-z][a-z0-9+.\-]*://[^\s:/@]+:)([^@\s/]{3,})@")),
    ("secret_kv", re.compile(
        r"(?i)\b(api[_-]?key|secret|token|password|passwd|pwd|access[_-]?key|client[_-]?secret)"
        r"(\s*[:=]\s*)(['\"]?)([^\s'\"]{6,})(\3)")),
]


def redact_secrets(text: str) -> tuple[str, list[str]]:
    found: list[str] = []
    for name, pat in _SECRETS:
        def _sub(m, _n=name):
            found.append(_n)
            if _n == "url_password":
                return m.group(1) + PLACE.format("secret") + "@"
            if _n == "secret_kv":
                return m.group(1) + m.group(2) + m.group(3) + PLACE.format("secret") + m.group(5)
            return PLACE.format("secret")
        text = pat.sub(_sub, text)
    return text, found


# ---- operator's own identifiers ---------------------------------------------
_MAC_RE = re.compile(r"\b(?:[0-9A-Fa-f]{2}[:-]){5}[0-9A-Fa-f]{2}\b")


@dataclass
class Profile:
    ips: set[str] = field(default_factory=set)          # your public + private IPs
    macs: set[str] = field(default_factory=set)
    hostnames: set[str] = field(default_factory=set)
    home_paths: set[str] = field(default_factory=set)
    usernames: set[str] = field(default_factory=set)
    device_ids: set[str] = field(default_factory=set)

    def merged(self, other: "Profile") -> "Profile":
        return Profile(*(a | b for a, b in zip(
            (self.ips, self.macs, self.hostnames, self.home_paths, self.usernames, self.device_ids),
            (other.ips, other.macs, other.hostnames, other.home_paths, other.usernames,
             other.device_ids))))


def _local_ips() -> set[str]:
    ips: set[str] = set()
    try:
        ips.add(socket.gethostbyname(socket.gethostname()))
    except OSError:
        pass
    try:  # standard trick to find the primary outbound-interface IP without sending a packet
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ips.add(s.getsockname()[0])
        s.close()
    except OSError:
        pass
    return {ip for ip in ips if ip and not ip.startswith("127.")}


def _local_mac() -> set[str]:
    node = uuid.getnode()
    if (node >> 40) & 0x01:            # locally-administered/random fallback bit -> not a real MAC
        return set()
    return {":".join(f"{(node >> (8 * i)) & 0xFF:02x}" for i in reversed(range(6)))}


def local_profile() -> Profile:
    """Best-effort identifiers of THIS machine, so the redactor knows what is 'yours'."""
    home = os.path.expanduser("~")
    host = socket.gethostname() or ""
    names = {host, host.split(".")[0]} if host else set()
    return Profile(ips=_local_ips(), macs=_local_mac(),
                   hostnames={n for n in names if len(n) >= 3},
                   home_paths={home} if home and home != "~" else set(),
                   usernames={u for u in {getpass.getuser()} if u and len(u) >= 3})


_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CONFIG = os.path.join(_HERE, "configs", "privacy.yaml")


def load_profile(config_path: str = _CONFIG) -> tuple[Profile, dict]:
    """Auto-detected local identifiers MERGED with configs/privacy.yaml (which holds what can't be
    auto-detected — your public IP above all). Returns (profile, toggles)."""
    prof, toggles = local_profile(), {"redact_secrets": True, "redact_identity": True,
                                      "enabled": True}
    try:
        import yaml
        with open(config_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        for k in toggles:
            if k in cfg:
                toggles[k] = bool(cfg[k])
        p = cfg.get("profile") or {}
        extra = Profile(ips=set(p.get("ips") or []), macs=set(p.get("macs") or []),
                        hostnames={h for h in (p.get("hostnames") or []) if len(str(h)) >= 3},
                        home_paths=set(p.get("home_paths") or []),
                        usernames={u for u in (p.get("usernames") or []) if len(str(u)) >= 3},
                        device_ids=set(str(d) for d in (p.get("device_ids") or [])))
        prof = prof.merged(extra)
    except (FileNotFoundError, ImportError, ValueError):
        pass
    return prof, toggles


def redact_identity(text: str, profile: Profile) -> tuple[str, list[str]]:
    found: list[str] = []
    # exact, longest-first literal identifiers (home path before username so the path wins)
    literal = ([(p, "home") for p in sorted(profile.home_paths, key=len, reverse=True)]
               + [(h, "hostname") for h in profile.hostnames]
               + [(i, "ip") for i in profile.ips]
               + [(m, "mac") for m in profile.macs]
               + [(d, "device_id") for d in profile.device_ids]
               + [(u, "user") for u in profile.usernames])
    for value, cat in sorted(literal, key=lambda kv: len(kv[0]), reverse=True):
        if not value:
            continue
        pat = re.compile(re.escape(value), re.I if cat in ("hostname", "user") else 0)
        text, n = pat.subn(PLACE.format(cat), text)
        found.extend([cat] * n)
    return text, found


# ---- combined ---------------------------------------------------------------
def redact(text: str, profile: Profile | None = None, *, secrets: bool = True,
           identity: bool = True) -> tuple[str, list[dict]]:
    """Clean `text` for an external boundary. Returns (redacted_text, findings). findings carry the
    category and count only — never the secret value."""
    if not text:
        return text or "", []
    all_found: list[str] = []
    if secrets:
        text, f = redact_secrets(text)
        all_found += f
    if identity and profile is not None:
        text, f = redact_identity(text, profile)
        all_found += f
    counts: dict[str, int] = {}
    for c in all_found:
        counts[c] = counts.get(c, 0) + 1
    findings = [{"category": c, "count": n, "placeholder": PLACE.format(c)}
                for c, n in sorted(counts.items())]
    return text, findings


if __name__ == "__main__":
    import sys
    src = sys.stdin.read() if not sys.stdin.isatty() else " ".join(sys.argv[1:])
    out, fnd = redact(src, local_profile())
    print(out)
    if fnd:
        print("\n[privacy] redacted: " + ", ".join(f"{f['count']}×{f['category']}" for f in fnd),
              file=sys.stderr)
