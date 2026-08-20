# A private, fine-tunable LLM — Moonlight-16B-A3B + QLoRA

Fine-tune an open-weight MoE model on free cloud GPUs, evaluate it honestly against a frozen
benchmark, and serve it from weights you control. No training on the laptop, no dependency on a
hosted API at inference.

**Status:** all four gates passed on a free Kaggle T4. The adapter trains, checkpoints, reloads
cold in a fresh process, and generates.

**Working base: Qwen2.5-Coder-14B-Instruct.** A held-out benchmark decided it over the original
Moonlight baseline — security capability 0.831 vs 0.787 (sec-v4, 68 items), equal web-bench and
prompt-injection resistance (both 1.000), and ~3× faster. Everything below about Moonlight remains
the **frozen baseline** (still loadable as `--model moonlight`); the security fine-tune now
specialises Qwen. The model is chosen in one place — `serving/model_spec.py` — via a lock file, so
the whole stack (training, serving, UI, benchmarks) follows `load_lock()` and nothing else hardcodes
a model name. See [`configs/qwen25_coder_qlora.yaml`](configs/qwen25_coder_qlora.yaml).

---

## What was selected, and why

| | |
|---|---|
| model | `moonshotai/Moonlight-16B-A3B-Instruct` |
| revision | `4e735b07a89f73647dfab71ab91b840f362ede5b` — pinned, never `main` |
| size | 15,960,111,936 params · 16B total / 3B active MoE |
| context | **8192** tokens |
| licence | MIT (declared in repo metadata; note there is **no LICENSE file** in the repo) |
| architecture | DeepSeek-V3, Multi-head Latent Attention, 64 routed + 2 shared experts |

**Why not Kimi K2.** K2 is 1T parameters. At 4-bit its weights are ~560GB, and MoE experts must
stay resident during training, so QLoRA needs 8×H100 minimum — about **$23/hour, or $555 for a
single 24-hour run**. Moonlight QLoRA runs on one free T4. That is the entire decision.

**What fine-tuning will and will not do.** LoRA adapts *behaviour* — format, register, refusal
patterns, tool-call style, domain vocabulary. It does not make a 16B model reason like a 1T one.
If you want K2-level capability you need K2 weights. Everything here is about control, not IQ.

---

## Three findings that would each have cost a GPU session

**1. The standard LoRA recipe silently fails on this model.**
`[q_proj, k_proj, v_proj, o_proj]` is the Llama default. This model uses Multi-head Latent
Attention and has **no `k_proj` and no `v_proj`**. The real targets are
`[q_proj, kv_a_proj_with_mqa, kv_b_proj, o_proj]`, verified against both the tensor index and a
live module tree. `scripts/freeze_spec.py` exits non-zero if a configured target does not exist.

**2. The model's own `modeling_deepseek.py` is broken on every installable transformers.**

```
transformers 5.15  ImportError: is_torch_fx_available          build fails
transformers 4.57  DynamicCache.get_usable_length missing      forward fails
transformers 4.48  from_legacy_cache assertion                 forward fails
transformers 4.44  no wheels for Python 3.13                   install fails
```

`deepseek_v3` is supported **natively** since transformers 4.49, so the fix is to stop using the
remote code: `trust_remote_code` is not needed for the model at all. (The tokenizer still needs
it — it is a custom tiktoken tokenizer.)

**2b. And "just upgrade" makes it worse in a way nothing reports as an error.**
transformers 5.0 refactored DeepSeek-V3 MoE experts from `ModuleList[Linear]` into 3D
`nn.Parameter` tensors. bitsandbytes quantises by *replacing `nn.Linear` modules*, so it cannot
reach them. Measured on this model with `scripts/check_quantizable.py`:

```
transformers 4.57.6    97.9% in nn.Linear     8.49 GB    fits a T4
transformers 5.15.0     7.7% in nn.Linear    30.08 GB    fits nothing free
```

Nothing raises. The module tree is correct and the LoRA targets still resolve — the model just
silently stops fitting. `train.py` refuses to start on 5.x rather than let you discover this
after a 32GB download. **4.57.6 is the newest 4.x release**, so this is the latest version that
works, not an old one. Everything else runs current: peft 0.20, accelerate 1.14, datasets 5.0.1,
torch 2.13.

**3. Attention-only vs attention+MLP is a 41× difference.**
At r=16: 7,105,536 trainable params vs 292,408,320. The MLP variant adapts 64 routed experts
across 26 layers, each adapter seeing only its routed fraction of tokens. Attention-only unless
the evaluation shows it underfitting.

---

## Repository

```
configs/moonlight_qlora.yaml     every value annotated with the constraint that set it
MODEL_SPEC.lock.json             pinned revision, licence, framework pins, verification record
scripts/
  freeze_spec.py                 pin + derive module tree + validate LoRA targets (~2MB download)
  smoke_test.py                  full training path on CPU: attach, fwd/bwd, save, reload
  kaggle_bootstrap.py            pre-flight; refuses a P100 before anything expensive
  verify_checkpoint.py           GATE 3 — cold reload in a fresh process, then generate
training/
  prepare_dataset.py             raw -> clean -> dedup -> filter -> tokenize -> split
  train.py                       QLoRA with four gates and auto-resume
evaluation/
  build_benchmark.py             builds and freezes the benchmark
  frozen/benchmark_v1/           120 items — NEVER trained against
  development/domain_expansion/  new cases go here instead
models/                          experiment-NNN/ per run; the base is never written to
```

---

## Running it

Kaggle: verify your phone (gates GPU **and** internet), then Accelerator → **GPU T4 x2**,
Internet → On.

```bash
python scripts/kaggle_bootstrap.py       # pre-flight; stops on a P100
python scripts/smoke_test.py             # ~1 min, no download
python training/train.py --pilot         # ~32GB download, then 10 steps + gates
python scripts/verify_checkpoint.py --checkpoint models/experiment-001/final
```

The bf16 download is ~32GB even though it loads as ~8.5GB in VRAM — bitsandbytes quantises after
fetching. Kaggle kills sessions at 9 hours, so do not start it and walk away.

**T4, not P100.** bitsandbytes 4-bit needs compute capability ≥ 7.5. T4 is 7.5, P100 is 6.0.
Kaggle assigns either; both the bootstrap and the trainer stop rather than fail confusingly.

---

## The four gates

The pilot exists to answer one question: *can the real model train, fit, checkpoint, resume and
generate?* It is **not** a quality test.

| gate | what it catches | result on a T4 |
|---|---|---|
| 1 · memory | peak VRAM per phase — load, forward, backward, optimizer, save | **13.67 GB peak, 1.97 GB spare** at 1024 tokens |
| 2 · loss | NaN, inf, exactly-flat, or divergent | 5.3311 → 3.9875, finite, not flat |
| 3 · reload | cold restart in a **fresh process** — in-process reloads share state and pass when a real restart would fail | loads cold in 496s, generates |
| 4 · provenance | revision, dataset hash, benchmark hash, config, package versions, CUDA, seeds | stamped into every checkpoint |

Throughput was **6.6 s/example** at 1024 tokens, so a 9-hour session fits roughly 4,800
example-passes. Most of that is padding: `padding="max_length"` pads every example to 1024 while
the seed data averages 107 content tokens. Dynamic padding is a large speedup available whenever
the memory guarantee is worth trading.

The pilot loss falling is **not** evidence of quality — 20 examples, cosine decay to zero across
10 steps. It proves gradients reach the adapters and nothing NaNs.

---

## Evaluation

120 items, frozen at `sha256 c370f8d0...`, **95 of them deterministically graded**.

```
coding 20 (real assertions)   reasoning 15        mathematics 15 (exact)
instruction following 10      technical 10        factuality 10 (false premises)
tool calling 10 (structural)  behaviour 10        trading/research 10
long-context/RAG 10 (synthetic contexts, exact ground truth, sized to 84% of the 8K window)
```

**Refusal is scored in both directions.** Six items are legitimate requests a badly-tuned model
wrongly declines; four should be declined. A model that answers everything scores ~50% — exactly
the same as one that refuses everything. Maximum score needs the boundary *and* a useful path
at it.

**Hallucination is tested with false premises** — a module, a flag and a paper that do not exist.
Correct behaviour is saying so.

`build_benchmark.py --verify` recomputes the hash and fails if the benchmark changed. Results
produced under different hashes are not comparable, and the tooling refuses to pretend otherwise.

### Running an evaluation

```bash
python evaluation/run_benchmark.py --base                              # once, then reused
python evaluation/run_benchmark.py --adapter models/experiment-004/final
python evaluation/compare.py --base evaluation/results/base \
                             --candidate evaluation/results/experiment-004
```

Base scores depend only on (revision, decode settings, benchmark hash), so they are measured
once and reused — re-running an 8-minute load to reproduce a constant wastes a quarter of a
Kaggle session. The cost of separate runs is drift, so every condition that could differ is
recorded and **compare.py exits non-zero rather than compare two runs measured differently**.
Decoding is greedy: a benchmark that samples cannot attribute a change to the model.

### Three tiers, not two

| tier | n | what it means |
|---|---|---|
| objective | 59 | a value matches, an assertion executes, a structure parses — reproducible to the character |
| rubric | 36 | keyword heuristics against prose criteria; directionally sound, noisy per item |
| judge | 25 | needs an LLM judge; **unscored** unless one is explicitly configured |

The objective row is the headline. Blending the three into one number lets rubric noise borrow
the credibility of executed assertions.

### Regression detection

A net gain is not an improvement. The report always prints improved / regressed / unchanged
together, flags any category dropping past 5%, and refuses a clean verdict when one has:

```
MIXED — objective +0.084, but 2 categories regressed past 5%:
    factuality  -0.130
    mathematics -0.050
A net gain with category regressions is a trade. Decide whether you want it.
```

`evaluation/test_harness.py` verifies all of this on CPU by planting a regression and confirming
the report catches it. A harness that cannot catch a deliberate regression cannot be trusted
with a real one.

---

## Local UI — a browser workstation for the agent

A localhost-only web interface to the agent, wired to the **real** stack — not a second
implementation. Chat drives the real `run_session` loop; tool calls surface as Approve/Deny cards and
run through the real controller only after you approve them; external text is sanitised by the trust
boundary; results are hashed into the telemetry ledger (the live activity feed); every answer is
graded by the verifier. No fake chatbot, no mock tools in the running app.

```bash
python start_local.py            # loads the real model on a GPU host; honest "gpu_required" on CPU
python start_local.py --stub     # clearly-labelled echo model to click through the UI without a GPU
```

Then open `http://127.0.0.1:8000`. The one reuse point is the injectable `generate(messages)->str`
seam that `run_session`/`run_assessment` already take, so the model code is unchanged. Binds
`127.0.0.1` only; the backend is authoritative (approval and the kill switch are enforced server-side,
tools stay gated by the session capability profile). Full documentation — architecture, security
notes, troubleshooting — is in **[`README_LOCAL_UI.md`](README_LOCAL_UI.md)**. Verified by
`tests/api/test_webui.py` (in the gate) and a live browser end-to-end run.

---

## Honest limitations

- **8K context.** Long documents go through RAG, not the prompt.
- **The pilot proves plumbing, not learning.** It trained on 20 synthetic seed examples. No
  claim about quality can be made until a real run is scored against the frozen benchmark.
- **Pinned to transformers 4.57.6 indefinitely.** Not by preference — 5.x cannot quantise the
  experts. If bitsandbytes gains 3D expert support this cap lifts; `check_quantizable.py` is the
  test for that.
- **35 → 25 benchmark items still need an LLM judge.** Judge-scored results are flagged so they
  can be excluded from headline numbers.
- **No LICENSE file in the upstream repo.** MIT is declared in metadata. That is normally
  accepted, but "declared MIT" is the accurate phrasing, not "ships an MIT licence".
- **The tokenizer requires `trust_remote_code`**, which executes code from the model repo. The
  model itself no longer does.
- **The local UI needs a GPU for the real model.** On a CPU laptop the model load is refused
  honestly (`gpu_required`) and the UI runs against a clearly-labelled `--stub`; the real-model
  conversation runs on a GPU host through the same lock seam.
