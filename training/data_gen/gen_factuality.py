"""gen_factuality.py — anti-hallucination training data with hard negatives.

The base run proved Moonlight confabulates: asked about a nonexistent module/flag/paper it
invents a confident API, and it states plain facts wrongly (16B active params instead of 3B,
self-attention instead of MLA). This targets that.

THE TRAP, AND THE FIVE BUCKETS THAT AVOID IT
Training only "fake -> deny, real -> answer" produces a model that denies anything it does not
instantly recognise — over-skepticism, which is just a different hallucination (falsely claiming
real things do not exist). So the set has five buckets, deliberately balanced:

  1 plausible real   -> answer confidently and correctly
  2 plausible fake   -> "I can't verify that; I won't invent its behaviour" + pointer to real
  3 obscure real     -> answer; do NOT falsely deny a real-but-lesser-known thing
  4 unknown          -> state calibrated uncertainty rather than guess
  5 false premise    -> challenge the wrong assumption embedded in the question

Buckets 3 and 4 are the anti-over-skepticism guard. Bucket 5 is the confidently-wrong-fact fix.

GROUND TRUTH
Every "real" and "obscure real" fact here was verified by execution or is a stable, checkable
fact. A wrong label trains the model to lie confidently, so this errs conservative. Fake
entities are disjoint from the frozen benchmark; prepare_dataset.py hard-fails on any overlap.
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

# 1 — PLAUSIBLE REAL: well-known, answer confidently. (name, kind, answer)
REAL = [
    ("torch.nn.Linear", "PyTorch class", "a fully-connected layer computing y = xA^T + b; "
     "constructed with in_features and out_features."),
    ("git rebase", "git command", "reapplies commits onto another base, producing linear history."),
    ("LoRA", "fine-tuning method", "Low-Rank Adaptation: trains small rank-r matrices added to "
     "frozen weights, updating a tiny fraction of parameters."),
    ("hashlib.sha256", "Python function", "returns a SHA-256 hash object; call .hexdigest() for "
     "the hex string."),
    ("HTTP 429", "status code", "Too Many Requests — the client is being rate-limited."),
    ("nmap -sV", "nmap flag", "enables service and version detection on open ports."),
    ("bitsandbytes 4-bit nf4", "quantization type", "4-bit NormalFloat quantization used for "
     "QLoRA via BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4')."),
    ("functools.lru_cache", "Python decorator", "memoizes a function's return values; maxsize "
     "bounds the cache, maxsize=None makes it unbounded."),
]

# 2 — PLAUSIBLE FAKE: does not exist; decline and redirect. (name, kind, real-nearby)
FAKE = [
    ("torch.nn.FastLinear", "PyTorch class", "torch.nn.Linear is the real fully-connected layer"),
    ("git supermerge", "git command", "you may be thinking of git merge or git rebase"),
    ("peft.QuantumLoRA", "PEFT class", "the real API is LoraConfig with get_peft_model"),
    ("hashlib.blake4", "Python function", "hashlib has blake2b and blake2s, not blake4"),
    ("HTTP 452", "status code", "452 is not standard; 429 and 503 cover rate-limit and outage"),
    ("nmap --auto-cve", "nmap flag", "nmap has -sV and the vuln NSE scripts, not --auto-cve"),
    ("bitsandbytes 3-bit nf3", "quantization type", "bitsandbytes offers 8-bit and 4-bit, not nf3"),
    ("the 2024 paper 'Gradient Teleportation in Sparse Transformers'", "paper",
     "I can find no real paper by that title"),
]

# 3 — OBSCURE REAL: real but lesser-known; must NOT be denied. (name, kind, answer) — all verified
OBSCURE_REAL = [
    ("itertools.pairwise", "Python function", "yes, added in Python 3.10; yields consecutive "
     "overlapping pairs from an iterable."),
    ("str.removeprefix", "Python method", "yes, added in Python 3.9; strips a prefix if present, "
     "otherwise returns the string unchanged."),
    ("math.isqrt", "Python function", "yes, since Python 3.8; the integer square root."),
    ("contextlib.suppress", "Python context manager", "yes; suppresses the given exceptions "
     "within its block."),
    ("HTTP 418", "status code", "yes, 'I'm a teapot' from RFC 2324 — a real, if joke, code."),
    ("socket SO_REUSEADDR", "socket option", "yes; allows a socket to bind an address in "
     "TIME_WAIT, commonly set before bind()."),
    ("git bisect", "git command", "yes; binary-searches commit history to find the one that "
     "introduced a bug."),
    ("secrets.compare_digest", "Python function", "yes; a constant-time comparison to avoid "
     "timing attacks on secrets."),
]

# 4 — UNKNOWN: might exist, not verifiable; state uncertainty rather than guess. (question, why)
UNKNOWN = [
    ("What exact learning rate did Moonshot use to pretrain Moonlight-16B-A3B?",
     "the precise pretraining hyperparameters are not something I can verify from public sources"),
    ("What is the full training-data composition of Moonlight-16B-A3B?",
     "the exact dataset mixture has not been publicly disclosed in a form I can verify"),
    ("How many GPUs did the Moonlight pretraining run use?",
     "I don't have a verified figure for that"),
    ("What is the internal API of the private 'apex-router' service?",
     "I have no way to verify the internals of a private service I can't see"),
    ("What did commit a1b2c3d in some private repo change?",
     "I can't inspect a commit in a repository I don't have access to"),
]

# 5 — FALSE PREMISE: the question embeds a wrong assumption; challenge it. (question, correction)
FALSE_PREMISE = [
    ("How do I use LoRA to update all of a model's weights during training?",
     "LoRA does the opposite: it freezes the base weights and trains small low-rank adapters. "
     "If you truly want to update every weight, that's full fine-tuning, not LoRA."),
    ("Since MD5 is collision-resistant, is it safe for deduplicating security-sensitive files?",
     "MD5 is NOT collision-resistant — practical collisions have existed since 2004. Use "
     "SHA-256 for anything security-sensitive."),
    ("Why does the GIL let Python threads run CPU-bound code in parallel across cores?",
     "It doesn't — the GIL prevents exactly that. CPU-bound parallelism in CPython needs "
     "multiprocessing or a native extension that releases the GIL."),
    ("How do I reverse a SHA-256 hash back to the original input?",
     "SHA-256 is a one-way function; there is no reverse. You can only brute-force or look up "
     "known inputs."),
    ("Since bitsandbytes 4-bit runs on any GPU, will it work on my compute-capability 6.0 card?",
     "It won't — 4-bit needs compute capability >= 7.5. A 6.0 card (e.g. P100) is below that."),
]

Q_ABOUT = ["What does {n} do?", "Can you explain {n}?", "How do I use {n}?",
           "Is {n} real? What is it?"]
A_REAL_T = ["Yes — {n} is {kind}: {ans}", "{n} is real ({kind}). {ans}"]
A_FAKE_T = [
    "I can't verify {n} as a real {kind}, so I won't invent its behaviour. {alt}. "
    "Where did you see it?",
    "I don't recognise {n} as a documented {kind}, and I won't fabricate one. {alt} — did you "
    "mean that?",
]
A_OBSCURE_T = ["{ans}", "Don't let the obscurity fool you — {ans}"]


def cap(s):
    return s[0].upper() + s[1:] if s else s


def ex(u, a):
    return {"messages": [{"role": "user", "content": u},
                         {"role": "assistant", "content": a}]}


def main():
    rng = random.Random(SEED)
    rows, by_bucket = [], {}

    def add(bucket, u, a):
        rows.append(ex(u, a))
        by_bucket[bucket] = by_bucket.get(bucket, 0) + 1

    for name, kind, ans in REAL:
        for q in Q_ABOUT[:3]:
            add("plausible_real", q.format(n=name),
                rng.choice(A_REAL_T).format(n=name, kind=kind, ans=ans))

    for name, kind, alt in FAKE:
        for q in Q_ABOUT:
            add("plausible_fake", q.format(n=name),
                rng.choice(A_FAKE_T).format(n=name, kind=kind, alt=cap(alt)))

    for name, kind, ans in OBSCURE_REAL:
        for q in (f"Does {name} exist?", f"Is {name} a real {kind}?", f"What is {name}?"):
            add("obscure_real", q, rng.choice(A_OBSCURE_T).format(ans=ans))

    for q, why in UNKNOWN:
        add("unknown", q, f"I don't actually know, and I won't guess: {why}. If you have a "
                          f"source, share it and I'll work from that.")

    for q, corr in FALSE_PREMISE:
        add("false_premise", q, f"That premise isn't right. {corr}")

    rng.shuffle(rows)
    out = os.path.join(HERE, "data", "raw", "factuality.jsonl")
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    print(f"wrote {len(rows)} factuality examples to {os.path.relpath(out, HERE)}")
    for b in ("plausible_real", "obscure_real", "plausible_fake", "unknown", "false_premise"):
        print(f"  {b:16} {by_bucket.get(b, 0)}")
    answer = by_bucket["plausible_real"] + by_bucket["obscure_real"]
    withhold = by_bucket["plausible_fake"] + by_bucket["unknown"] + by_bucket["false_premise"]
    print(f"  ── answer/challenge {answer} : withhold/deny {withhold}  "
          f"(both sides, so it learns the distinction not denial)")


if __name__ == "__main__":
    main()
