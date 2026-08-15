"""relevance.py — gate retrieved passages by ACTUAL relevance before grounding on them.

Held-out Security Bench v3 proved the failure this closes: on an XXE question the retriever
returned web_application_security.md (XSS/CSRF/SQLi — NOT XXE) at 0.319, just over the 0.30 floor.
The model grounded on it and produced the DOC's vocabulary ("parameterized queries", "CSP", "XML
injection") instead of XXE/external-entity reasoning — turning a correct base answer (1.00 without
RAG) into a wrong one (0.17 with RAG).

Why not just raise the score floor? Because cosine cannot separate relevant from tangential at this
granularity: v2 incident-response NEEDED its doc at 0.358 (grounding helped 0.25->1.00), while v3
XXE was DRAGGED by a doc at 0.319. Those overlap; no single threshold splits them. The distinction
is semantic — does the passage actually contain information for THIS question — so the gate is a
semantic judgment, not a number.

    keep = gate_relevance(question, hits, judge_fn)   # judge_fn(prompt)->text (the model)

judge_fn is the model itself (base Moonlight is strong at this yes/no, and it costs one short
generation). Deferring to the model's own knowledge when nothing survives the gate is SAFE and
measured: v3 showed the base answers held-out security correctly on its own (1.000). So the gate
FAILS CLOSED — if the judge's reply can't be parsed, keep nothing and let the model answer from
knowledge, rather than risk grounding on junk.

Testable on CPU: judge_fn is a callable, so gate_relevance's parse/filter logic is unit-tested with
a scripted judge (relevance_test.py). The judgment QUALITY needs the real model, on the GPU.
"""
from __future__ import annotations

import re

RELEVANCE_PROMPT = (
    "You are filtering retrieved passages for relevance before they are used to answer a question. "
    "A passage is RELEVANT only if it contains information that DIRECTLY helps answer THIS specific "
    "question. A passage about a related but DIFFERENT topic (same broad field, different subject) "
    "is NOT relevant, even if it shares keywords.\n\n"
    "Question: {question}\n\n"
    "Passages:\n{passages}\n\n"
    "Reply with ONLY the numbers of the directly-relevant passages, comma-separated (e.g. '1,3'), "
    "or the single word NONE if none of them directly help. Do not explain."
)

PASSAGE_CHARS = 400          # enough to judge topic without spending the whole context


def build_relevance_prompt(question: str, hits: list[dict]) -> str:
    listing = "\n".join(f"[{i + 1}] {(h.get('text') or '')[:PASSAGE_CHARS]}"
                        for i, h in enumerate(hits))
    return RELEVANCE_PROMPT.format(question=question, passages=listing)


def parse_relevant_indices(reply: str, n: int) -> list[int]:
    """0-based indices of passages the judge kept. Empty on NONE or on anything unparseable
    (FAIL CLOSED: when in doubt, ground on nothing and let the model use its own knowledge)."""
    if not reply:
        return []
    low = reply.lower()
    nums = {int(x) for x in re.findall(r"\d+", reply) if 1 <= int(x) <= n}
    if not nums:
        return []                       # NONE, or no valid number -> keep nothing
    # guard against a model that echoes "none" AND stray digits
    if "none" in low and not re.search(r"\b([1-9]\d*)\b", re.sub(r"none", "", low)):
        return []
    return sorted(i - 1 for i in nums)


def gate_relevance(question: str, hits: list[dict], judge_fn) -> list[dict]:
    """Return only the hits that genuinely help answer `question`. May be empty (-> defer to the
    model's own knowledge via query.NO_CONTEXT). Adds one short judge generation per call."""
    if not hits:
        return []
    reply = judge_fn(build_relevance_prompt(question, hits)) or ""
    keep = parse_relevant_indices(reply, len(hits))
    return [hits[i] for i in keep]
