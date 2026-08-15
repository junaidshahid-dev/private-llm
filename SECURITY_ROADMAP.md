# Security as a defining strength — the plan, honestly

The goal: make this assistant *exceptionally* strong at cybersecurity and ethical hacking, not
merely conversant. Security is a first-class capability with its own measurement, its own data,
and its own improvement loop — separate from general capability.

This document is deliberately honest about what is claimed vs measured, because the failure mode
here is confident hype. **No claim of "better than existing models" is made without a number
against a named baseline.**

## The three layers (never conflated)

```
KNOWLEDGE      broad security reasoning — networking, Linux/Windows internals, web, authn/z,
               vuln research, exploit-dev concepts, RE, malware/binary analysis, privesc, AD,
               cloud, mobile, API, crypto, wireless, forensics, detection, IR, secure coding,
               methodology, and reading Nmap/Burp/Wireshark/Metasploit/Ghidra/debugger/ADB output.
               Lives in the model + RAG. Grown by data and retrieval, measured by the benchmark.

AUTHORIZATION  what the operator has approved to test. Explicit list, class-agnostic, expiring.
               (mcp_layer/security.py) — knowledge is never narrowed to fit authorization.

EXECUTION      operator-directed only. The model proposes; the operator runs. Enforced in code.
               (mcp_layer/controller.py) — reasoning cannot become action.
```

## How capability is measured, not asserted

`evaluation/development/security_capability/` is a benchmark that tests **reasoning over
evidence** — given a scan result, a request, a code snippet, a log, an APK manifest, does the
model reach the correct technical conclusion? — not definition recall. It is scored separately
from the general benchmark, so security is tracked on its own axis.

The protocol, every round:

1. run the security benchmark against **base Moonlight** and **strong open baselines** (e.g. a
   larger open model where it fits, and a general model of similar size) under identical decode
   settings.
2. record per-domain scores. Improvement is a delta against those baselines, never a vibe.
3. freeze a benchmark version once it is broad enough to anchor comparisons; new items go to a
   development split so the frozen score stays comparable over time.

Seed today: 12 items, one per domain. This grows toward dozens per domain, with an LLM judge for
the reasoning-quality items (flagged `judge_recommended`) kept separate from the deterministic
signal.

## The reasoning workflow the benchmark is built to reward

A strong security assistant should, on a real problem:
understand the environment → ask for missing detail → hypothesise → select tools → interpret
output → correlate across sources → explain the mechanism → suggest methodology → adapt on
failure → conclude precisely. The `controller.plan → operator executes → controller.interpret`
loop is built around exactly this, with the operator holding every trigger.

## How it gets better (progressive, cheapest-first)

Given the measured result that small-scale SFT *degrades* this model, capability is grown in this
order — the expensive, risky step last:

1. **RAG corpus** (now): index authoritative security material — your notes, CVE/CWE references,
   tool docs, methodology, past engagement writeups (never model outputs). Highest leverage, zero
   capability risk. This is where most near-term gains come from.
2. **Tool interpretation**: feed real Nmap/tshark/APK/log output back through `interpret()`; the
   model reasons over ground truth instead of guessing.
3. **Benchmark growth**: more items per domain, so the score is trustworthy and improvements are
   visible.
4. **A stronger base model**: when budget allows, use the strongest open-weight model that fits
   the hardware, and re-measure on the same benchmark. This is the biggest lever and the most
   expensive; take it when the eval says the current base is the ceiling.
5. **Targeted fine-tuning** — only if a much larger, length-balanced, reasoning-rich security
   dataset exists, and only if the benchmark shows it helps without the regressions two SFT rounds
   already produced. Earned by evidence, not assumed.

## What "the strongest model realistically available for our budget" means today

Moonlight-16B-A3B, because it is the only Kimi-family model that runs QLoRA/4-bit on a free T4
(measured: 97.9% quantisable, 8.5GB; the next step up needs paid GPUs). When the budget moves,
step 4 above re-opens the model choice — and the benchmark decides, not enthusiasm.
