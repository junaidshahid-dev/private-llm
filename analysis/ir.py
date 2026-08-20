"""ir.py — incident-response helpers: IOC extraction, log timeline, tamper/anomaly signs. Pure.

Read-only text analysis for the IR workflow: pull indicators of compromise out of logs/artifacts,
build a chronological timeline, and flag signs of log tampering. Everything is OBSERVED text — an
extracted "domain" or a burst of failed logins is a LEAD to corroborate, never proof. Log content is
untrusted data (route it through the trust boundary before a model reads it).
"""
from __future__ import annotations

import re

# low-false-positive indicators. 'domain' is restricted to common TLDs so 'file.txt'/'2.4.25' don't
# masquerade as domains.
_IOC = {
    "ipv4": re.compile(r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b"),
    "url": re.compile(r"\bhttps?://[^\s'\"<>]+", re.I),
    "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    "domain": re.compile(r"\b(?:[a-z0-9-]+\.)+(?:com|net|org|io|gov|edu|co|ru|cn|info|biz|xyz|onion|"
                         r"local|internal|dev|app|cloud)\b", re.I),
    "sha256": re.compile(r"\b[a-fA-F0-9]{64}\b"),
    "sha1": re.compile(r"\b[a-fA-F0-9]{40}\b"),
    "md5": re.compile(r"\b[a-fA-F0-9]{32}\b"),
    "cve": re.compile(r"\bCVE-\d{4}-\d{4,7}\b", re.I),
    "btc": re.compile(r"\b(?:bc1|[13])[a-km-zA-HJ-NP-Z1-9]{25,39}\b"),
}
_ISO_TS = re.compile(r"\b(\d{4}-\d{2}-\d{2})[T ](\d{2}:\d{2}:\d{2})")
_SYSLOG_TS = re.compile(r"\b([A-Z][a-z]{2})\s+(\d{1,2})\s+(\d{2}:\d{2}:\d{2})\b")
_TAMPER = [
    (r"(?i)\b(?:log|logs|history)\s+cleared\b", "logs cleared"),
    (r"(?i)\bwtmp\b|\butmp\b|\blastlog\b", "login-record file touched"),
    (r"(?i)\bunset\s+HISTFILE\b|\bhistory\s+-c\b", "shell history disabled/cleared"),
    (r"(?i)\bjournalctl\s+--rotate\b|\btruncate\b.*log", "journal rotated/truncated"),
    (r"(?i)\brm\b\s+.*/var/log", "log files deleted"),
]


def extract_iocs(text: str) -> dict:
    """Indicators of compromise grouped by type (deduped, sorted). Leads to corroborate, not proof."""
    text = text or ""
    out = {}
    for k, rx in _IOC.items():
        vals = sorted({m.group(0) for m in rx.finditer(text)})
        if vals:
            out[k] = vals
    return out


def log_timeline(text: str, limit: int = 200) -> list[dict]:
    """Extract timestamped log lines into a chronological list of {ts, line} (ISO + syslog formats)."""
    events = []
    for line in (text or "").splitlines():
        m = _ISO_TS.search(line)
        if m:
            events.append({"ts": f"{m.group(1)}T{m.group(2)}", "sortable": True,
                           "line": line.strip()[:200]})
            continue
        s = _SYSLOG_TS.search(line)
        if s:
            events.append({"ts": f"{s.group(1)} {int(s.group(2)):02d} {s.group(3)}",
                           "sortable": False, "line": line.strip()[:200]})
    events.sort(key=lambda e: (not e["sortable"], e["ts"]))
    return events[:limit]


def log_anomalies(text: str) -> list[dict]:
    """Signs worth investigating: log-tampering markers + auth-failure bursts."""
    text = text or ""
    out = []
    for pat, name in _TAMPER:
        if re.search(pat, text):
            out.append({"sign": "tampering", "detail": name})
    fails = len(re.findall(r"(?i)\b(authentication failure|failed password|invalid user|"
                           r"access denied|401 |403 )\b", text))
    if fails >= 5:
        out.append({"sign": "auth_failures", "detail": f"{fails} authentication-failure indicators "
                    "(possible brute force / spray) — corroborate with source IPs + timing"})
    return out


def analyze_log(text: str) -> dict:
    ev = log_timeline(text)
    return {"events": len(ev), "timeline": ev[:50], "iocs": extract_iocs(text),
            "anomalies": log_anomalies(text),
            "note": "OBSERVED log text; IOCs and anomalies are leads to corroborate, not proof"}
