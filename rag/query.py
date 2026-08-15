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

GROUNDED_SYSTEM = (
    "Answer the question using ONLY the numbered context below. "
    "If the context does not contain the answer, say you do not have that information — do not "
    "use outside knowledge and do not guess. When you use a fact, cite its [n]. Be accurate and "
    "as detailed as the context supports."
)
NO_CONTEXT = (
    "No relevant context was found in the knowledge base for this question. Tell the user you do "
    "not have information on this in your documents, rather than answering from memory."
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
