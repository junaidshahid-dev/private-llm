# RAG — grounded, honest answers without touching the model

This is the factuality fix we arrived at after measuring that small-scale fine-tuning *degraded*
Moonlight. Instead of trying to bake facts into weights, retrieve them from your own documents at
inference and make the model answer only from that context. Facts you can correct in seconds by
editing a file; hallucinations replaced by honest "I don't have that" when the corpus is silent.

**Everything except generation runs on CPU.** Local embeddings (all-MiniLM-L6-v2), a numpy index,
no data leaves the machine, no GPU or API needed to build or search.

## Use it

```bash
# 1. build an index from your documents (notes, code, research, docs)
python rag/ingest.py path/to/your/docs

# 2. see what it retrieves + the assembled prompt (CPU only)
python rag/query.py "your question" --show

# 3. full answer with base Moonlight (needs the GPU — Kaggle)
python rag/answer.py "your question"
```

## How it stays honest

- **Grounded prompt:** the model is told to answer *only* from the retrieved context and cite it.
- **Abstention:** if the best match scores below `MIN_SCORE` (0.30 cosine), the prompt tells the
  model to say it has nothing on the topic — instead of answering from memory. This is what
  turns "invent a fake module" into "I don't have information on that."
- **Corpus hygiene:** model outputs/logs (`results/`) and index directories are never indexed —
  RAG is only as trustworthy as what you feed it (indexing the model's own hallucinations was a
  real bug we hit and fixed).

## Verified on the exact failures the base model had

| question | base Moonlight | with RAG |
|---|---|---|
| DeepSeek-V3 attention? | "self-attention" (wrong) | retrieves MLA fact → correct |
| `mcp.server.turbo_fastmcp`? | invents a whole module | scores 0.22 < 0.30 → abstains |

## Files

```
store.py     the index: local embed, save/load, cosine search
ingest.py    documents -> chunks -> index (skips model outputs and index dirs)
query.py     retrieve + build the grounded/abstaining prompt (CPU)
answer.py    retrieve + policy layer + generate with base Moonlight (GPU)
knowledge/   the curated knowledge base (edit these; re-run ingest)
```

The index (`rag/index/`) is a build artifact and is gitignored — regenerate it from your docs.
