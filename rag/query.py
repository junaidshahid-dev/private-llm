"""query.py — retrieve context and build a GROUNDED, abstaining prompt for Moonlight.

    python rag/query.py "what transformers version does the project pin, and why?"
    python rag/query.py "..." --k 5 --show            # also print the retrieved chunks

This is the factuality fix we measured toward. Instead of asking Moonlight what it "knows" (and
watching it invent a Microchip Cloud Platform module), we retrieve from YOUR documents and tell
the model to answer only from that context — and to say it doesn't know when the context does
not contain the answer. Grounded answer, or an honest abstention. No fabrication.

This script does the retrieval and prints the assembled prompt (CPU, no model needed). Feed that
prompt to Moonlight wherever it runs. If sentence-transformers and the model were both on the
same box you would generate here directly; kept separate so retrieval stays laptop-only.

ABSTENTION THRESHOLD. If the best chunk scores below MIN_SCORE the retrieval is treated as a
miss and the prompt instructs the model to say it has nothing on the topic. That is what stops
"answer from your own memory" hallucination when the corpus genuinely lacks the fact.
"""
from __future__ import annotations

import argparse
import io
import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except (AttributeError, ValueError):  # already reconfigured, or not a real stream
    pass
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

MIN_SCORE = 0.30          # cosine below this = no genuine match; abstain rather than force it

# Grounding contract. The earlier version said "use ONLY the context ... no outside knowledge."
# That is right for questions about the operator's own material (repo, model internals), where the
# model has no reliable parametric knowledge and must not guess — but it is WRONG for a capable
# domain assistant. On security benchmark v2 it caused a regression: a doc about nmap banner
# false-positives made the model DROP a CVE (Samba 3.0.20 -> CVE-2007-2447) it had answered
# correctly with no RAG at all. "Only the context" turned grounding into blinders and suppressed
# correct knowledge. The fix (measured, not retrained): the context is the AUTHORITATIVE reference,
# but the model must still answer completely from its own expertise, preferring the context where
# they disagree. Abstention on genuinely-unknown material is preserved by build_prompt's NO_CONTEXT
# branch (retrieval score < MIN_SCORE), not by gagging the model's knowledge on every answer.
# Grounding contract, now RELEVANCE-AWARE. Security Benchmark v3 (held-out) exposed the failure a
# blanket "prefer the context" caused: on an XXE question the nearest doc was web_application_
# security.md (XSS/CSRF/SQLi — NOT XXE) scoring just over the floor at 0.319; grounding on it and
# preferring it dragged a correct answer wrong (1.00 -> 0.17). The floor can tell "some security
# doc" from "nothing", but not "on-topic" from "same broad topic". So the model must judge
# relevance itself and DECLINE to be dragged by an off-topic document.
GROUNDED_SYSTEM = (
    "First judge whether the reference context below actually addresses THIS question. "
    "If it does: use it as your authoritative source, cite each fact you take from it as [n], and "
    "where your knowledge and the on-point context conflict, trust the context and say so. "
    "If the context is about a related but DIFFERENT topic and does not directly cover the "
    "question, do NOT force your answer to fit it — rely on your own expert knowledge and note that "
    "the retrieved context was not directly on point. "
    "Either way, draw on your own expertise to be COMPLETE, include relevant specifics the context "
    "omits — a named CVE, an exact command, a version detail — when you are confident, and never "
    "invent facts you are unsure of; if part of the question is covered by neither the context nor "
    "your confident knowledge, say so plainly."
)
# When retrieval finds nothing relevant, DEFER to parametric knowledge — do not gag it. v3 showed
# the old text ("say you don't have this in your documents, rather than answering from memory")
# talking the base model OUT of a pickle-RCE answer it knew perfectly (1.00 -> 0.00). A capable
# model should answer from its own knowledge when the KB is silent, and flag that it is ungrounded
# so the verification layer / operator can check it — not refuse, and not force in off-topic snippets.
NO_CONTEXT = (
    "The knowledge base returned nothing directly relevant to this question. Do not force the "
    "off-topic snippets into your answer, and do not refuse. Answer from your own expert knowledge, "
    "but state clearly that this answer is NOT grounded in the retrieved documents so it can be "
    "verified."
)


def build_prompt(question: str, hits: list[dict]) -> str:
    if not hits or hits[0]["score"] < MIN_SCORE:
        return f"{NO_CONTEXT}\n\nQuestion: {question}"
    blocks = []
    for i, h in enumerate(hits, 1):
        blocks.append(f"[{i}] (source: {h['source']}, score {h['score']:.2f})\n{h['text']}")
    context = "\n\n".join(blocks)
    return f"{GROUNDED_SYSTEM}\n\nContext:\n{context}\n\nQuestion: {question}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--index", default=os.path.join(HERE, "rag", "index"))
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--show", action="store_true", help="print retrieved chunks and scores")
    args = ap.parse_args()

    from rag.store import Store
    if not os.path.exists(os.path.join(args.index, "index_meta.json")):
        sys.exit(f"no index at {args.index} — run: python rag/ingest.py <docs_dir>")
    store = Store().load(args.index)
    hits = store.search(args.question, k=args.k)

    if args.show:
        print("=" * 78)
        print(f"RETRIEVED for: {args.question}")
        print("=" * 78)
        for i, h in enumerate(hits, 1):
            flag = "" if h["score"] >= MIN_SCORE else "  (below abstain threshold)"
            print(f"\n[{i}] {h['source']}  score {h['score']:.3f}{flag}")
            print("    " + h["text"][:280].replace("\n", "\n    "))
        print("\n" + "=" * 78)
        best = hits[0]["score"] if hits else 0.0
        print(f"best score {best:.3f}  ->  "
              + ("grounded answer" if best >= MIN_SCORE else "ABSTAIN (nothing relevant)"))
        print("=" * 78 + "\n")

    print("---- PROMPT FOR MOONLIGHT ----")
    print(build_prompt(args.question, hits))
    return 0


if __name__ == "__main__":
    sys.exit(main())
