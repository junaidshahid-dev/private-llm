"""telemetry.py — the full auditable chain of an assessment (spec #20).

Records every step so a finding is traceable end to end: user instruction -> model plan -> proposed
action -> authorization decision -> tool invoked + target -> OUTPUT HASH (sha256 of the raw result,
so the interpretation can be checked against what the tool actually returned) -> sanitized result
preview -> model interpretation -> verification verdict -> report.

Two disciplines: secrets are REDACTED before anything is stored (privacy.redact), and every tool
result is HASHED so results are tamper-evident and a fabricated interpretation cannot silently claim
output the tool never produced. In-memory by default; pass a `sink` callable to persist (e.g. JSONL).
"""
from __future__ import annotations

import hashlib
import json
import time

from privacy.redact import redact_secrets


def output_hash(data) -> str:
    """sha256 of a tool result — a stable fingerprint of what the tool actually returned."""
    if isinstance(data, (bytes, bytearray)):
        b = bytes(data)
    else:
        b = json.dumps(data, sort_keys=True, default=str, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(b).hexdigest()


def _clean(v):
    if isinstance(v, str):
        text, _ = redact_secrets(v)
        return text
    return v


class Telemetry:
    def __init__(self, session_id: str = "session", sink=None):
        self.session_id = session_id
        self.records: list = []
        self.sink = sink                              # optional callable(record) to persist

    def record(self, kind: str, **fields) -> dict:
        rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "session": self.session_id, "kind": kind}
        for k, v in fields.items():
            rec[k] = _clean(v)
        self.records.append(rec)
        if self.sink:
            try:
                self.sink(rec)
            except Exception:                         # noqa: BLE001 — telemetry never breaks the run
                pass
        return rec

    # ---- convenience recorders for each step of the chain -----------------------------------
    def instruction(self, text: str):
        return self.record("instruction", text=(text or "")[:2000])

    def plan(self, text: str):
        return self.record("plan", text=(text or "")[:4000])

    def proposal(self, tool: str, arguments: dict, why: str = ""):
        return self.record("proposal", tool=tool, arguments={k: _clean(v) for k, v in
                                                              (arguments or {}).items()},
                           why=(why or "")[:300])

    def authorization(self, tool: str, target: str, allowed: bool, reason: str = ""):
        return self.record("authorization", tool=tool, target=(target or ""), allowed=bool(allowed),
                           reason=(reason or "")[:300])

    def tool_result(self, tool: str, target: str, ok: bool, output) -> dict:
        return self.record("tool_result", tool=tool, target=(target or ""), ok=bool(ok),
                           output_sha256=output_hash(output),
                           output_preview=_clean(str(output))[:400])

    def interpretation(self, text: str):
        return self.record("interpretation", text=(text or "")[:4000])

    def verdict(self, verdict: str, findings=None):
        return self.record("verification", verdict=verdict, findings=list(findings or [])[:20])

    def report(self, summary: str):
        return self.record("report", summary=(summary or "")[:1000])

    # ---- read back --------------------------------------------------------------------------
    def chain(self) -> list:
        return list(self.records)

    def kinds(self) -> list:
        return [r["kind"] for r in self.records]

    def render(self) -> str:
        L = [f"TELEMETRY — session {self.session_id} ({len(self.records)} events)"]
        for r in self.records:
            extra = {k: v for k, v in r.items() if k not in ("ts", "session", "kind")}
            L.append(f"  {r['ts']} {r['kind']}: " + json.dumps(extra, default=str)[:160])
        return "\n".join(L)
