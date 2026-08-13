# Phase 1 — Model selection report

**Researched 2026-08-13.** Prices and model availability move weekly; re-check before committing
money. Every figure below is sourced, not assumed.

---

## The Kimi / Moonshot open-weight lineup

| model | total params | active | context | licence | notes |
|---|---|---|---|---|---|
| **Kimi K3** | 2.8T | — | 1M | open weights (Jul 2026) | newest, largest |
| **Kimi K2.6 / K2.7 Code / K2 Thinking / K2.5** | **1T** | 32B | 262k | **Modified MIT** | flagship family, MoE |
| **Kimi-VL-A3B** | ~16B | 2.8B | — | MIT | vision-language |
| **Moonlight-16B-A3B-Instruct** | **16B** | **3B** | **8K** | **MIT** | MoE, DeepSeek-V3 architecture |

There is a very large hole in the middle of that table. Moonshot ships a ~16B class and a 1T
class, and nothing between them.

---

## The cost calculation — this is the whole decision

Current rental prices (RunPod, Aug 2026): RTX 4090 24GB **$0.34/hr** community, RTX A6000 48GB
**$0.49/hr**, A100 80GB **$1.39/hr**, H100 PCIe **$2.89/hr**. Vast.ai spot ~$0.35/hr with no SLA.

### Kimi K2 (1T params) — QLoRA

MoE weights must be resident; you cannot stream experts cheaply during training.

```
weights at 4-bit                    ~500-560 GB
+ activations, adapter grads/optim  ~600 GB+ VRAM needed
minimum viable                      8x H100 80GB = 640 GB   (very tight)
realistic                           8x H200 141GB
```

```
8 x H100 PCIe @ $2.89/hr        =  $23.12 / hour
one 24-hour LoRA run            =  $555
weight download                 =  ~560 GB
persistent storage for weights  =  ~$28-56 / month, idle
```

**One training run costs roughly six times your entire account balance.** Experimentation —
which is most of what fine-tuning actually is — multiplies that. Kimi K3 at 2.8T is ~3x worse.

### Moonlight-16B-A3B — QLoRA

```
weights at 4-bit                    ~9 GB
QLoRA on a single A6000 48GB        comfortable
also fits a 24GB 4090 at short seq lengths
```

```
A6000 @ $0.49/hr, 8-hour run    =  $3.92
A100 80GB @ $1.39/hr, 8 hours   =  $11.12   (faster, more headroom)
weight download                 =  ~32 GB
```

**$4 versus $555.** That is the entire decision, and nothing else in this project changes it.

---

## Recommendation

**Moonlight-16B-A3B-Instruct**, on the following grounds:

- **MIT licence.** Not "Modified MIT" — plain MIT. Commercial use, redistribution and
  fine-tuning are all permitted without restriction. Requirement 12 satisfied cleanly.
- **DeepSeek-V3 architecture**, so vLLM, SGLang and transformers all support it already. No
  custom inference stack, no waiting for support. Phases 10 and 12 become straightforward.
- **3B active parameters** means fast, cheap inference — an A6000 at $0.49/hr serves it
  comfortably, and you can shut it down between sessions.
- Fine-tuning is affordable enough to **iterate**, which is what actually produces a good
  result. One expensive run you cannot repeat is worth less than twenty cheap ones.

---

## What you must know before agreeing

**1. Fine-tuning does not add intelligence.** This is the single most misunderstood thing about
this whole project. LoRA/QLoRA adapts *behaviour* — format, tone, refusal patterns, domain
vocabulary, tool-call style. It does not make a 16B model reason like a 1T model. If you want
K2-level reasoning you need K2 weights, and those cost $555 per run to touch.

**2. Moonlight's context is 8K tokens.** Your requirements list long-context analysis. 8K is
short — roughly 6,000 words, in and out combined. RAG (Phase 12) mitigates this by retrieving
only what's relevant, and that is the correct design anyway. But you will not be doing
whole-repository analysis in one prompt on this model.

**3. The honest capability expectation.** Moonlight-16B-A3B is an early-2025 small MoE. It will
be clearly weaker than the frontier assistants you use daily, at everything. What you gain is
control: your weights, your inference stack, your system prompt, no API dependency, no external
logging. That is a real and legitimate thing to want. It is not a capability upgrade.

**4. On refusal behaviour.** With an MIT-licensed open base model this is a legitimate
configuration problem, not a bypass: system prompt, sampling parameters, and light SFT on your
own examples. Nothing here involves attacking a third-party service, and that boundary is
respected throughout the design.

---

## Where the money actually goes

```
model download + storage        ~$2-5 / month
QLoRA experiments (10 runs)     ~$40-110
evaluation runs                 ~$5-10
inference, 20 hrs/month         ~$10 on an A6000
                                --------
first month, realistic          ~$60-130
```

Against a $91.86 balance that is tight but not absurd — **if** the GPU is stopped whenever it is
not training or serving. An instance left running by accident at $1.39/hr costs $33 a day.
Phase 14's cost controls are not optional here; they are what keeps this project from
consuming the account.

---

## Decision needed before Phase 2

**A.** Moonlight-16B-A3B — buildable now, ~$4/run, real control, modest capability, 8K context.
**B.** Kimi K2 — genuine frontier capability, ~$555/run, needs roughly 6x your current balance
for a single experiment.
**C.** Neither, if the honest capability expectation in point 3 isn't what you wanted.

There is no version of B that fits the current budget. If B is the goal, the prerequisite is
income, not architecture.
