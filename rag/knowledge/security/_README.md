# Security knowledge base

Authoritative reference material the assistant grounds its security answers in. RAG retrieves from
here, so **accuracy is the whole game** — wrong content here becomes a confident wrong answer
(the garbage-in-garbage-out lesson from indexing model outputs). Every doc is written to be
correct and general.

## Source tracking (principle 2: versioned, tracked)

Each file carries a header:

```
> source: curated-reference | version: v1 | updated: 2026-08-14 | authored
```

`authored` means written as a correct reference summary, not copied from an authoritative external
source. That is the honest label: this is a solid, correct *seed*, not the final corpus. It should
be **grown and eventually superseded** by real authoritative sources — OWASP cheat sheets, RFCs,
vendor docs, the CVE/CWE databases, tool man pages, papers — ingested with a license check. When
you add those, keep the source/version header so provenance stays traceable.

## Measurement honesty

These docs cover the same domains the security benchmark tests, so RAG *will* help on the
benchmark. That is the real effect we want to measure — "does having a knowledge base improve
reasoning" — but it is only fair because the docs are **general reference**, not the benchmark's
specific scenarios or answers. Do not write KB content tailored to benchmark items; that would
measure "I gave it the answers", not retrieval. The benchmark lives in `evaluation/` and is
excluded from ingestion.

## Growing it

```bash
python rag/ingest.py rag/knowledge        # re-index after adding/editing docs
python rag/query.py "your question" --show
```

Add your own material here too: your projects, notes, code, past engagement writeups (never model
outputs or logs). The more correct, relevant, source-tracked knowledge, the more capable and
honest the assistant becomes — without touching the model.
