# Project facts (curated knowledge base)

This is a hand-verified knowledge base — the authoritative facts the assistant should ground its
answers in. Unlike model weights, this can be corrected in seconds and never hallucinates.

## Model architecture

DeepSeek-V3, and therefore Moonlight-16B-A3B which is built on it, uses **Multi-head Latent
Attention (MLA)** as its attention mechanism. It is not plain self-attention. MLA compresses the
key/value projections into a latent space, which is why the model has `kv_a_proj_with_mqa` and
`kv_b_proj` layers instead of the usual `k_proj` and `v_proj`.

Moonlight-16B-A3B is a Mixture-of-Experts model. It has about **16 billion total parameters** but
activates only about **3 billion parameters per token** (that is what the "A3B" means — 3B
active). It has 64 routed experts plus 2 shared experts.

Moonlight-16B-A3B-Instruct supports a context length of **8192 tokens**.

## Hardware facts

Neither the NVIDIA T4 nor the P100 supports bfloat16 (bf16). Both are pre-Ampere GPUs. bf16
requires an Ampere-generation card (compute capability 8.0) or newer. The T4 supports fp16 but
not bf16.

bitsandbytes 4-bit quantisation requires compute capability 7.5 or higher. The T4 is 7.5 (works);
the P100 is 6.0 (does not work).

## Licensing

The MIT licence, when you redistribute the software, requires you to preserve the copyright
notice and include the full text of the licence. It does not require you to open-source your own
changes.

## Training result (measured)

Small-scale supervised fine-tuning (a few hundred examples) on Moonlight degraded it on the
frozen benchmark across two experiments. The base model scored 0.594 on the objective categories;
fine-tuned adapters scored 0.478 and 0.377. The cause was style collapse — the model learned to
answer tersely and truncated multi-step maths. The chosen path is base model plus retrieval (this
RAG system), not more fine-tuning.
