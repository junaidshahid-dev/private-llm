"""controls.py — operator memory commands over the store (Phase 11.5).

The user-facing verbs: remember / recall / forget / correct / what_do_you_remember / show_sources.
Thin, deliberate wrappers over MemoryStore so the assistant (or a UI) exposes a clear, reversible
interface. Every command is explicit and auditable; nothing edits memory silently.
"""
from __future__ import annotations


def remember(store, text, mtype="semantic", importance=0.6, **kw):
    """remember X — store a durable fact (refused if it contains a secret)."""
    return store.add(text, mtype=mtype, importance=importance, **kw)


def recall(store, query, k=5):
    """what do you remember about X — relevance-gated retrieval."""
    return store.search(query, k=k)


def forget(store, target):
    """forget X — archive by id if it is one, else archive everything matching the query."""
    if store.get(target):
        return store.forget(target)
    return store.forget_matching(target)


def correct(store, query, new_text):
    """correct/replace X — supersede the best-matching memory (old kept as history)."""
    hits = store.search(query, k=1)
    if not hits:
        return {"ok": False, "error": f"no memory matches {query!r} to correct"}
    return store.supersede(hits[0]["id"], new_text)


def what_do_you_remember(store, query=None, k=10):
    """List remembered items (optionally about a topic)."""
    if query:
        return store.search(query, k=k)
    return store._active(store.project)[:k]


def show_sources(store, query, k=5):
    """Where a memory came from — provenance for 'why do you believe this'."""
    return [{"text": m["text"], "source": m["source"], "type": m["type"],
             "confidence": m["confidence"], "created": m["created"]}
            for m in store.search(query, k=k)]
