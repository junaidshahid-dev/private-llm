# Moonlight baseline (frozen)

This is the reference the model-swap experiment measures against. **Do not change the stack to
improve it** — the point is to hold everything constant except the model and see what a stronger
brain buys. Frozen at git tag `moonlight-baseline`.

## The frozen stack

```
Moonlight-16B-A3B-Instruct (4-bit)
  + RAG (relevance-gated retrieval)
  + MCP tools (read-only + gated security tools)
  + explicit operator approval (propose -> approve -> execute)
  + verification (math/code/grounding/phantom-action/tool-fabrication/CVE/hash)
  + controlled security lab (DVWA, isolated)
  + benchmarks (security v2 keyword-rubric, v3 held-out semantic + deterministic anchors)
```

Model pinned in `MODEL_SPEC.lock.json`: `moonshotai/Moonlight-16B-A3B-Instruct` @
`4e735b07a89f73647dfab71ab91b840f362ede5b`.

## What this baseline measured (honest)

- **Security v3 (held-out, self-judge):** base 1.000, base+RAG 0.975 after the relevance gate.
  The 1.000 means the 10/11-item self-judged benchmark is *saturated*, not that the model is perfect
  — read the delta and the review queue, never the absolute.
- **RAG:** helps where the model lacks knowledge, *hurt* out-of-domain until the relevance gate;
  now at parity. RAG's real value is niche/current/project-specific facts, not general security.
- **Live lab workflow:** `nmap` + `ffuf` live-tested against DVWA; Moonlight interprets real recon
  with the analyst discipline (evidence vs inference, banner-is-not-proof, impact ranking) and does
  **not** fabricate (verification PASS is legitimate). Remaining weakness = deeper security
  reasoning + tool selection (it proposed a redundant re-scan) — captured as benchmark item
  `v3_toolselect_03`, **not** as a prompt rule.

## The model-swap experiment

The architecture is model-agnostic: nothing but `MODEL_SPEC.lock.json` names a model, and the
runners resolve it via `serving/model_spec.py:load_lock()`. To test a stronger open-weight model
against this baseline **without unfreezing it**:

1. Write a second lock file, e.g. `MODEL_SPEC.<name>.lock.json`, same shape as the baseline
   (`model`, `revision`, `context_length`, `quantization`). Confirm it is quantisable on the target
   GPU (`scripts/check_quantizable.py`).
2. Run the **exact same** workflow, pointing at it via the env var — the baseline file is untouched:
   ```bash
   MODEL_LOCK=MODEL_SPEC.<name>.lock.json python evaluation/run_secv3.py
   MODEL_LOCK=MODEL_SPEC.<name>.lock.json python evaluation/run_secv3.py --rag rag/index
   MODEL_LOCK=MODEL_SPEC.<name>.lock.json python bridge/interpret_recon.py
   ```
3. Compare to the baseline runs (same RAG, same MCP, same verification, same lab, same benchmark).

That answers one clean question: **does a stronger brain improve security reasoning when everything
else is identical?** — and keeps all the work already done.

### Candidate model constraints
Must be quantisable to 4-bit and fit the run GPU (single T4 ≈ 16 GB after the CUDA_VISIBLE_DEVICES=0
pin), open-weight, and stronger at reasoning/security than a 16B/3B-active MoE. Pick, write a lock,
`check_quantizable.py`, then run the three commands above.
