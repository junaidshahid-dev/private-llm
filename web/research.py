"""research.py — multi-step research mode: plan -> search -> read -> synthesize -> cite -> verify.

Turns the four web tools into a researcher. The model generates focused queries; web_search finds
candidates; web_extract reads the top few into an evidence set; the model synthesizes an answer that
CITES its sources [n], flags contradictions, and says plainly when the evidence is insufficient
(never bluffs). Verification then checks the answer against the fetched evidence.

Read-only over the PUBLIC web (web_search + web_extract are both read-only and SSRF-gated), so the
pipeline runs once the operator has enabled `web` — the approval is at "enable web + run research",
like RAG retrieval, not per page. Every fact is attributable to a source URL (source="web").

`generate`, `searcher`, and `extractor` are injectable, so the whole orchestration is unit-tested
with mocks; live research (real model + network) is validated in the end-of-phase runs.
"""
from __future__ import annotations

import re

MAX_EVIDENCE_CHARS = 2500          # per source, kept in the synthesis prompt
PLAN_SYSTEM = ("You are planning web research. Output up to {n} focused search queries, one per "
               "line, no numbering or prose — just the queries.")
SYNTH_SYSTEM = (
    "Answer the question using ONLY the numbered web evidence below. Cite every fact as [n]. If the "
    "evidence does NOT contain the answer, say so plainly — do not guess or use outside knowledge. "
    "If sources CONTRADICT each other, say so and show both. Attribute claims to their source URL.")


def plan_queries(question: str, generate, max_q: int = 3) -> list[str]:
    raw = generate([{"role": "system", "content": PLAN_SYSTEM.format(n=max_q)},
                    {"role": "user", "content": question}]) or ""
    qs = [re.sub(r"^\s*(?:\d+[.)]|[-*])\s*", "", ln).strip() for ln in raw.splitlines()]
    qs = [q for q in qs if q and len(q) > 2][:max_q]
    return qs or [question]


def gather_evidence(queries, config, k, searcher, extractor) -> list[dict]:
    seen, candidates = set(), []
    for q in queries:
        r = searcher(config, q)
        if r.get("ok"):
            for c in r["result"]["results"]:
                u = c.get("url", "")
                if u and u not in seen:
                    seen.add(u)
                    candidates.append(c)
    evidence = []
    for c in candidates:
        if len(evidence) >= k:
            break
        e = extractor(config, c["url"])
        if e.get("ok"):
            evidence.append({"url": e["result"]["final_url"], "title": c.get("title", ""),
                             "text": (e["result"]["text"] or "")[:MAX_EVIDENCE_CHARS]})
    return evidence


def synthesize(question, evidence, generate) -> str:
    blocks = "\n\n".join(f"[{i+1}] {e['title']} — {e['url']}\n{e['text']}"
                         for i, e in enumerate(evidence))
    return (generate([{"role": "system", "content": SYNTH_SYSTEM},
                      {"role": "user", "content": f"Question: {question}\n\nWeb evidence:\n{blocks}\n\n"
                       "Answer with citations; flag contradictions; say if evidence is insufficient."}])
            or "").strip()


def research(question, generate, config, *, searcher=None, extractor=None, k_sources=3,
             max_queries=3, verify_fn=None) -> dict:
    if searcher is None:
        from web.search import web_search
        searcher = web_search
    if extractor is None:
        from web.extract import web_extract
        extractor = web_extract
    if verify_fn is None:
        from verification.verify import verify as verify_fn

    queries = plan_queries(question, generate, max_queries)
    evidence = gather_evidence(queries, config, k_sources, searcher, extractor)

    if not evidence:
        answer = ("Insufficient web evidence: no sources could be retrieved for this question. "
                  "I will not guess.")
        sufficient = False
    else:
        answer = synthesize(question, evidence, generate)
        sufficient = True

    hits = [{"source": e["url"], "text": e["text"], "score": 1.0} for e in evidence]
    report = verify_fn(answer, hits=hits or None)
    return {"question": question, "queries": queries,
            "sources": [{"url": e["url"], "title": e["title"]} for e in evidence],
            "evidence_count": len(evidence), "sufficient": sufficient, "answer": answer,
            "verification": {"verdict": report.verdict,
                             "findings": [str(f) for f in report.findings]}}


def render(rec: dict) -> str:
    L = [f"RESEARCH: {rec['question']}", f"queries: {rec['queries']}",
         f"sources ({rec['evidence_count']}):"]
    for i, s in enumerate(rec["sources"], 1):
        L.append(f"  [{i}] {s['title']} — {s['url']}")
    if not rec["sufficient"]:
        L.append("  (no evidence retrieved)")
    L.append("\nANSWER:\n" + rec["answer"])
    L.append(f"\nVerification: {rec['verification']['verdict']}")
    for f in rec["verification"]["findings"]:
        L.append(f"  - {f}")
    return "\n".join(L)
