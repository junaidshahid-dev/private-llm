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


_IMPORTANT = ("decided", "decision", "prefer", "always", "never", "must", "architecture", "chose",
              "standard", "policy", "rule", "north star", "baseline", "uses", "directory", "goal")
_TEMPORARY = ("today", "right now", "currently", "temporar", "might", "maybe", "for now", "later",
              "tomorrow", "just now", "at the moment")


def classify_importance(text: str) -> float:
    """Heuristic importance in [0,1]: decisions/preferences/architecture score higher; temporary or
    hedged statements score lower. Replaces the old fixed default."""
    lo = (text or "").lower()
    score = 0.5
    if any(w in lo for w in _IMPORTANT):
        score += 0.25
    if any(w in lo for w in _TEMPORARY):
        score -= 0.3
    if len(text) < 20:
        score -= 0.1
    return round(max(0.0, min(1.0, score)), 2)


def _terms(text: str) -> set:
    return {t for t in re.findall(r"\w+", (text or "").lower()) if len(t) > 3}


def _is_duplicate(text: str, store, project) -> bool:
    terms = _terms(text)
    if not terms:
        return False
    return any(len(terms & set(re.findall(r"\w+", i["text"].lower()))) / len(terms) >= 0.8
               for i in store._active(project or store.project))


def detect_conflicts(text: str, store, project=None) -> list:
    """Existing memories that likely CONTRADICT the candidate: same subject (strong term overlap)
    but not a duplicate — e.g. 'base model is Moonlight' vs 'base model is Qwen'. These are surfaced
    for the operator to resolve (supersede), never blindly stored alongside the old fact."""
    terms = _terms(text)
    if not terms:
        return []
    out = []
    for i in store._active(project or store.project):
        overlap = len(terms & _terms(i["text"])) / len(terms)
        if 0.5 <= overlap < 0.8:
            out.append(i)
    return out


def process_candidates(candidates, store, project=None, approver=None,
                       importance=None, source="conversation") -> dict:
    """Filter + store candidates. Pipeline: secret-drop -> duplicate-skip -> CONFLICT-flag ->
    approval-gate -> store (importance CLASSIFIED unless given). approver(text)->bool is the operator
    gate. Conflicts are surfaced (not blindly stored), so contradictions are resolved deliberately.
    Returns a full account of every candidate (nothing hidden)."""
    added, dropped_secret, duplicate, declined, conflicts = [], [], [], [], []
    for text in candidates:
        _clean, secrets = redact_secrets(text)
        if secrets:
            dropped_secret.append(text)
            continue
        if _is_duplicate(text, store, project):
            duplicate.append(text)
            continue
        conf = detect_conflicts(text, store, project)
        if conf:
            conflicts.append({"candidate": text, "conflicts_with": [c["id"] for c in conf],
                              "existing": [c["text"] for c in conf]})
            continue                              # do NOT store both as current — surface it
        if approver is not None and approver(text) is not True:
            declined.append(text)
            continue
        imp = importance if importance is not None else classify_importance(text)
        r = store.add(text, importance=imp, project=project, source=source)
        (added if r.get("ok") else dropped_secret).append(text)
    return {"added": added, "duplicate": duplicate, "dropped_secret": dropped_secret,
            "declined": declined, "conflicts": conflicts}


def remember_from_conversation(conversation, generate, store, project=None, approver=None) -> dict:
    cands = extract_candidates(conversation, generate)
    result = process_candidates(cands, store, project=project, approver=approver)
    result["candidates"] = cands
    return result
