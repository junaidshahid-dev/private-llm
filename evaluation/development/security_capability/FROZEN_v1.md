# Security Benchmark v1 — FROZEN, historical

**Do not modify. Do not use for model-vs-model or before/after comparisons.**

v1 (`seccap.jsonl`, 20 items) uses a low-discrimination grader: it credits an answer if any single
expected keyword appears anywhere in the response. Base Moonlight scored **0.90** on it.

That 0.90 is **"usually reaches the right conclusion-word," not "90% security ability."** A shallow
answer that name-drops the right term passes; the two "failures" were the grader missing correct
answers phrased differently. It measures fluency, not correctness or completeness.

It is kept only as a historical marker of where measurement started. All real security measurement
uses **Security Benchmark v2** (`../security_v2/`), which requires multiple necessary components
per answer, penalises harmful hallucinations, and separates shallow from deep reasoning (proven:
shallow 0.2 vs deep 1.0 on the same item). The `Moonlight Security Baseline v1 = 0.90` result stays
committed at `evaluation/results/seccap_base_v1/` with this same caveat in its report.
