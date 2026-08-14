"""gen_factuality.py — training data for the anti-hallucination objective.

The base run proved Moonlight confabulates: asked about a module, flag, or paper that does not
exist, it invents a confident API rather than saying it cannot verify it. This generates
examples that target that, and writes data/raw/factuality.jsonl in the seed format.

THE CENTRAL RISK, AND HOW THIS AVOIDS IT
The naive fix — train it to say "that doesn't exist" — produces the opposite failure: a model
that denies REAL things too, which would fail the real-entity items in the benchmark and be
useless. So this set is deliberately BALANCED:

    REAL entities  -> confident, correct, specific answer
    FAKE entities  -> "I can't verify that as a real X; I won't invent its behaviour",
                      plus a genuine pointer to what IS real nearby

The lesson is the DISTINCTION and calibrated uncertainty, not denial. A model that learns to
say "I don't recognise that" to everything has not improved.

CONTAMINATION
None of the fake entities here reuse the benchmark's (turbo_fastmcp, --deterministic-moe, the
"Recursive Attention Collapse" paper). They are a disjoint set, and prepare_dataset.py now hard-
fails on any exact overlap with the frozen benchmark or the dev sets.
"""
from __future__ import annotations

import io
import json
import os
import random
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace",
                              line_buffering=True)
HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SEED = 20260814

# --- REAL entities: (name, kind, correct one-line answer) ------------------------------------
# Deliberately well-known and stable, so the "confident answer" is actually correct.
REAL = [
    ("torch.nn.Linear", "PyTorch class", "a fully-connected layer applying y = xA^T + b; you "
     "give it in_features and out_features."),
    ("transformers.AutoModelForCausalLM", "transformers class", "the auto-class that loads a "
     "causal (decoder-only) language model from a checkpoint via from_pretrained."),
    ("git rebase", "git command", "reapplies commits from one branch onto another, rewriting "
     "history to produce a linear sequence."),
    ("bitsandbytes 4-bit (nf4)", "quantization type", "a 4-bit NormalFloat quantization used by "
     "BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4') for QLoRA."),
    ("LoRA", "fine-tuning method", "Low-Rank Adaptation: trains small rank-r matrices added to "
     "frozen weights, so only a tiny fraction of parameters is updated."),
    ("python -m venv", "python command", "creates an isolated virtual environment in the given "
     "directory."),
    ("hashlib.sha256", "python function", "returns a SHA-256 hash object; call .hexdigest() for "
     "the hex string."),
    ("HTTP 429", "status code", "Too Many Requests — the client has sent too many requests in a "
     "given time (rate limiting)."),
    ("nmap -sV", "nmap flag", "enables service/version detection on open ports."),
    ("SELECT ... WHERE", "SQL clause", "filters rows in a query by a boolean condition."),
]

# --- FAKE entities: (name, kind, the-real-thing-nearby) ---------------------------------------
# Plausible-sounding but not real. The third field is a genuine pointer offered in the denial, so
# the model learns to redirect helpfully rather than just refuse. NONE reuse benchmark items.
FAKE = [
    ("torch.nn.FastLinear", "PyTorch class", "torch.nn.Linear is the real fully-connected layer"),
    ("transformers.AutoModelForTurboCausalLM", "transformers class",
     "the real class is AutoModelForCausalLM"),
    ("git supermerge", "git command", "you may be thinking of git merge or git rebase"),
    ("bitsandbytes 3-bit (nf3)", "quantization type",
     "bitsandbytes offers 8-bit and 4-bit (nf4/fp4), not a 3-bit nf3"),
    ("peft.QuantumLoRA", "PEFT class", "the real classes are LoraConfig and get_peft_model"),
    ("python --turbo-gc", "python flag", "there is no such flag; garbage collection is tuned via "
     "the gc module"),
    ("hashlib.blake4", "python function", "hashlib has blake2b and blake2s, not blake4"),
    ("HTTP 452", "status code", "452 is not a standard status code; 429 and 503 cover rate/limit "
     "and unavailability"),
    ("nmap --auto-cve", "nmap flag", "nmap has -sV and the vuln NSE scripts, not an --auto-cve "
     "flag"),
    ("SQL MERGEINTO*", "SQL clause", "standard SQL has MERGE; the exact syntax varies by engine"),
    ("accelerate.HyperTrainer", "accelerate class", "accelerate provides Accelerator, not a "
     "HyperTrainer"),
    ("the 2024 paper 'Gradient Teleportation in Sparse Transformers'", "paper",
     "I can't find a real paper by that title"),
]

Q_REAL = [
    "What does {n} do?",
    "Can you explain {n}?",
    "How do I use {n}?",
    "Is {n} a real thing, and what is it?",
]
Q_FAKE = [
    "What does {n} do?",
    "Explain the {n} feature.",
    "How do I use {n}?",
    "Does {n} exist? What does it do?",
    "Give me the API for {n}.",
]

A_REAL = [
    "Yes. {n} is {kind}: {ans}",
    "{n} is real — it's {kind}. {ans}",
]
A_FAKE = [
    "I can't verify {n} as a real {kind}, so I won't invent its behaviour. If you meant "
    "something similar, {alt}. Can you point me to where you saw it?",
    "I don't recognise {n} as a documented {kind}, and I shouldn't fabricate one. {alt} — "
    "is that what you're after?",
    "That doesn't look real to me: I can find no {kind} called {n}. Rather than make something "
    "up, I'll flag it — {alt}.",
]


def cap(s):
    return s[0].upper() + s[1:] if s else s


def main():
    rng = random.Random(SEED)
    rows = []

    for name, kind, ans in REAL:
        for q in Q_REAL:
            a = rng.choice(A_REAL).format(n=name, kind=kind, ans=ans)
            rows.append({"messages": [{"role": "user", "content": q.format(n=name)},
                                      {"role": "assistant", "content": a}]})

    for name, kind, alt in FAKE:
        for q in Q_FAKE:
            a = rng.choice(A_FAKE).format(n=name, kind=kind, alt=cap(alt))
            rows.append({"messages": [{"role": "user", "content": q.format(n=name)},
                                      {"role": "assistant", "content": a}]})

    rng.shuffle(rows)
    out = os.path.join(HERE, "data", "raw", "factuality.jsonl")
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    n_real = len(REAL) * len(Q_REAL)
    n_fake = len(FAKE) * len(Q_FAKE)
    print(f"wrote {len(rows)} factuality examples to {os.path.relpath(out, HERE)}")
    print(f"  real-entity (answer confidently)  {n_real}")
    print(f"  fake-entity (decline to invent)   {n_fake}")
    print(f"  balance real:fake = {n_real}:{n_fake} — teaches the DISTINCTION, not denial")


if __name__ == "__main__":
    main()
