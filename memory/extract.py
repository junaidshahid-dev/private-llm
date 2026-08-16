"""extract.py — turn a conversation into durable memories, safely. (Phase 11.2)

Do NOT save everything. The pipeline: the model proposes candidate long-term facts; then we FILTER
them — drop anything containing a secret, drop duplicates of what we already know, and (optionally)
require the operator's approval — before anything is stored. Temporary/trivial statements are meant
to be dropped by the extraction prompt; secrets and duplicates are dropped structurally here.

Conflict RESOLUTION (a new fact superseding an old one) stays operator-driven via controls.correct()
/ store.supersede() — auto-merging memories is risky, so extraction only ADDS genuinely-new facts.

`generate` is injectable, so the whole pipeline is unit-tested with a scripted model.
"""
from __future__ import annotations

import re

from privacy.redact import redact_secrets

EXTRACT_SYSTEM = (
    "Extract DURABLE facts worth remembering long-term from the conversation: the user's stable "
    "preferences, decisions made, and project facts. One fact per line, plain statements, no "
    "numbering. SKIP anything temporary, trivial, or uncertain, and SKIP secrets/credentials. If "
    "there is nothing worth remembering, output exactly: NONE")


def extract_candidates(conversation: str, generate, max_candidates: int = 8) -> list[str]:
    raw = generate([{"role": "system", "content": EXTRACT_SYSTEM},
                    {"role": "user", "content": conversation}]) or ""
    if raw.strip().upper().startswith("NONE"):
        return []
    out = []
    for ln in raw.splitlines():
        ln = re.sub(r"^\s*(?:\d+[.)]|[-*])\s*", "", ln).strip()
        if len(ln) >= 8 and ln.upper() != "NONE":
            out.append(ln)
    return out[:max_candidates]


def _is_duplicate(text: str, store, project) -> bool:
    terms = {t for t in re.findall(r"\w+", text.lower()) if len(t) > 3}
    if not terms:
        return False
    for i in store._active(project or store.project):
        existing = set(re.findall(r"\w+", i["text"].lower()))
        if len(terms & existing) / len(terms) >= 0.8:
            return True
    return False


def process_candidates(candidates, store, project=None, approver=None,
                       importance=0.6, source="conversation") -> dict:
    """Filter + store candidates. approver(text)->bool gates each one (operator approval). Returns a
    full account of what happened to every candidate (nothing hidden)."""
    added, dropped_secret, duplicate, declined = [], [], [], []
    for text in candidates:
        _clean, secrets = redact_secrets(text)
        if secrets:
            dropped_secret.append(text)
            continue
        if _is_duplicate(text, store, project):
            duplicate.append(text)
            continue
        if approver is not None and approver(text) is not True:
            declined.append(text)
            continue
        r = store.add(text, importance=importance, project=project, source=source)
        (added if r.get("ok") else dropped_secret).append(text)
    return {"added": added, "duplicate": duplicate, "dropped_secret": dropped_secret,
            "declined": declined}


def remember_from_conversation(conversation, generate, store, project=None, approver=None) -> dict:
    cands = extract_candidates(conversation, generate)
    result = process_candidates(cands, store, project=project, approver=approver)
    result["candidates"] = cands
    return result
