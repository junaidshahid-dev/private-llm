"""ingest.py — turn a directory of documents into a searchable index.

    python rag/ingest.py <docs_dir> [--out rag/index] [--model ...]

Reads text-like files (.md .txt .py .json .rst .yaml .toml and similar), splits them into
overlapping chunks on natural boundaries, embeds them locally, and saves the index. Re-run it
whenever your documents change — the index is a build artifact, not something you edit by hand.

CHUNKING. Split on blank lines (paragraphs / code blocks), then greedily pack paragraphs up to a
character budget with a one-paragraph overlap so an answer that straddles a boundary is still
retrievable. A single oversized paragraph is split on sentence boundaries. Each chunk keeps its
source file and position so answers can cite where the fact came from.
"""
from __future__ import annotations

import argparse
import io
import os
import re
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except (AttributeError, ValueError):  # already reconfigured, or not a real stream
    pass
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

TEXT_EXT = {".md", ".txt", ".py", ".json", ".jsonl", ".rst", ".yaml", ".yml", ".toml",
            ".cfg", ".ini", ".csv", ".html", ".js", ".ts", ".sh", ".sql"}
# "results" holds the model's OWN generated outputs — including its hallucinations. Indexing
# those would let RAG retrieve a fabrication as if it were a fact (garbage in, garbage out).
# "evaluation" holds the benchmarks: indexing them would let the model RETRIEVE the benchmark
# answers, invalidating every score. Neither is authoritative knowledge; both are skipped so the
# security benchmark can never become RAG (or training) data.
SKIP_DIRS = {".git", ".venv", "__pycache__", "node_modules", ".cache", "hf_cache", "models",
             "results", "evaluation"}
MAX_CHARS = 1200          # ~ 250-300 tokens, comfortably under MiniLM's window
OVERLAP_PARAS = 1


def split_paragraphs(text: str) -> list[str]:
    paras = re.split(r"\n\s*\n", text)
    out = []
    for p in paras:
        p = p.strip()
        if not p:
            continue
        if len(p) <= MAX_CHARS:
            out.append(p)
        else:                                    # oversized: split on sentence boundaries
            buf = ""
            for sent in re.split(r"(?<=[.!?])\s+", p):
                if len(buf) + len(sent) + 1 > MAX_CHARS and buf:
                    out.append(buf.strip())
                    buf = sent
                else:
                    buf = f"{buf} {sent}".strip()
            if buf:
                out.append(buf.strip())
    return out


def chunk_text(text: str, source: str) -> list[dict]:
    paras = split_paragraphs(text)
    chunks, i = [], 0
    while i < len(paras):
        buf, j = [], i
        while j < len(paras) and sum(len(p) for p in buf) + len(paras[j]) <= MAX_CHARS:
            buf.append(paras[j])
            j += 1
        if not buf:                              # single paragraph already at the cap
            buf, j = [paras[i]], i + 1
        chunks.append({"text": "\n\n".join(buf), "source": source,
                       "para_start": i, "para_end": j - 1})
        if j >= len(paras):
            break
        i = max(j - OVERLAP_PARAS, i + 1)        # step back for overlap, always advance
    return chunks


def read_dir(docs_dir: str) -> list[dict]:
    chunks = []
    for root, dirs, files in os.walk(docs_dir):
        # Never index a RAG index directory (it contains embeddings + the chunk text), or
        # re-ingesting a folder that holds its own output silently doubles and poisons the corpus.
        if "embeddings.npy" in files or "index_meta.json" in files:
            dirs[:] = []
            continue
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for fn in sorted(files):
            if os.path.splitext(fn)[1].lower() not in TEXT_EXT:
                continue
            path = os.path.join(root, fn)
            try:
                text = open(path, encoding="utf-8", errors="replace").read()
            except OSError:
                continue
            if not text.strip():
                continue
            rel = os.path.relpath(path, docs_dir)
            chunks.extend(chunk_text(text, rel))
    return chunks


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("docs_dir", help="directory of documents to index")
    ap.add_argument("--out", default=os.path.join(HERE, "rag", "index"))
    ap.add_argument("--model", default=None)
    args = ap.parse_args()

    from rag.store import Store, DEFAULT_MODEL
    docs = args.docs_dir if os.path.isabs(args.docs_dir) else os.path.join(os.getcwd(),
                                                                           args.docs_dir)
    if not os.path.isdir(docs):
        sys.exit(f"not a directory: {docs}")

    print(f"reading {docs} ...")
    chunks = read_dir(docs)
    if not chunks:
        sys.exit("no indexable text found (checked extensions: "
                 + ", ".join(sorted(TEXT_EXT)) + ")")
    n_src = len({c["source"] for c in chunks})
    print(f"  {len(chunks)} chunks from {n_src} files")

    store = Store(args.model or DEFAULT_MODEL)
    print(f"embedding with {store.model_name} (first run downloads ~80MB) ...")
    store.build(chunks)
    store.save(args.out)
    print(f"saved index -> {os.path.relpath(args.out, HERE)}  "
          f"({len(chunks)} vectors, dim {store.emb.shape[1]})")
    print(f"\nquery it:  python rag/query.py \"your question\"")
    return 0


if __name__ == "__main__":
    sys.exit(main())
