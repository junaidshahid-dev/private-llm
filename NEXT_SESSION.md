# NEXT SESSION — what to run, and where

Everything buildable on CPU is **done, tested, and committed**. What remains needs a **GPU (Kaggle)**
or a **residential IP / lab**, so it can't run in the build environment. This file is the exact,
verified command list. Every command below was checked against the real CLI signatures — no guesses.

---

## 0. Status at handoff (already verified locally, no GPU)

| Thing | State |
|---|---|
| CPU regression (`scripts/run_tests.py`) | **25/25 modules PASS** |
| Memory: encryption at rest, rollback, importance classifier, conflict detection, project seed | done + `features_test.py` (14 checks) in the gate |
| Memory Benchmark v1 | 7/7 |
| Web Benchmark v1 (fixtures) | PASS |
| Assessment state graph, verification layer, operator loop | PASS |
| **`web_fetch` live** | ✅ verified against a real public URL (200, UNTRUSTED-labeled, redirect chain) |
| **`web_search` live** | ⚠️ code correct + unit-tested, but DuckDuckGo returns **HTTP 202 + empty** to datacenter IPs. Now reported **loudly** (`backend_blocked`), not as "0 results". Needs a residential IP or a keyed search API. |

**Training queued: NONE.** We do **not** retrain — small-scale SFT measurably degraded Moonlight
**twice** (objective 0.594→0.478→0.377, math −0.533). Capability is added at inference (RAG +
behaviour policy), not by fine-tuning. Real training only returns at **Phase 13 (own model)**, which
is far off. **Phase 12 is a model *swap* (a lock-file change), not a training run.**

---

## 1. KAGGLE (GPU / T4) — the Phase-12 decision blocker: cross-judge

**Why:** the self-judge has home-field bias (it rated Moonlight 1.000 vs Qwen 0.795, while the
unbiased deterministic grader said 0.675 vs 0.645 — a tie). The decision needs an **independent
judge**: author ≠ judge. One model fits in VRAM at a time, so we generate both authors' answers
first, then judge each with the *other* model. The **deterministic overall** is judge-independent and
is the real anchor; the cross-judge is the second opinion with bias removed.

> Kaggle: Accelerator = **GPU T4**, Internet = **ON** (for the model download only — the web
> benchmark ships its own fixtures). If it assigns a **P100**, restart to reroll — P100 can't run
> bitsandbytes 4-bit. `kaggle_bootstrap.py` checks this before anything expensive.

### Cell 1 — setup
```bash
!git clone https://github.com/junaidshahid-dev/private-llm.git /kaggle/working/LLM
%cd /kaggle/working/LLM
!python scripts/kaggle_bootstrap.py
```

### Cell 2 — generate both authors' answers (self-judged; overridden below)
```bash
!python evaluation/run_secv3.py --model moonlight
!python evaluation/run_secv3.py --model qwen
```
Produces `evaluation/results/secv3_moonlight_base/` and `evaluation/results/secv3_qwen25-coder-14b_base/`.

### Cell 3 — CROSS-JUDGE, both directions (author ≠ judge)
```bash
!python evaluation/judge_pass.py secv3_moonlight_base        --judge-model qwen
!python evaluation/judge_pass.py secv3_qwen25-coder-14b_base --judge-model moonlight
```
Each prints **CROSS-JUDGED overall** and **deterministic overall (unbiased)**. Read those four
numbers — that is the verdict data. (Also saved to `secv3_<author>_judgedby_<judge>/`.)

### Cell 4 — web benchmark, both models (GPU; fixtures, no internet needed)
```bash
!python web/run_webbench.py --model moonlight
!python web/run_webbench.py --model qwen
```
> Note: the web benchmark now has **3 prompt-injection items** (direct, role-hijack, base64) and the
> web layer routes all retrieved content through the trust boundary (web/trust.py: detect → defang →
> envelope + a hardened system prompt). Both models previously **failed** injection with the old soft
> label; this run re-measures `injection_resisted` with the real fix in place. Item count is now 7, so
> the overall is not comparable to the earlier 0.700/0.800 — read the `injection_resisted` line.

### Cell 5 — optional side-by-side table + keep the results
```bash
!python evaluation/compare_v3.py moonlight qwen25-coder-14b
```

### Cell 6 — the methodology-scaffolding probe (does Qwen stop jumping to exploit?)
Both models interpret the SAME real recon (Apache 2.4.25 + discovered high-value paths) through the
identical `SECURITY_METHODOLOGY` scaffolding, trust boundary, and verification. Run each, then compare:
```bash
!python bridge/methodology_experiment.py --model moonlight
!python bridge/methodology_experiment.py --model qwen
!python bridge/methodology_experiment.py --compare
```
Read the `--compare` DECISION line. Rule: never switch on speed; Qwen becomes a Phase-12 candidate
only if it does NOT jump from the banner to exploitation and shows no material regression vs Moonlight
on tool_selection / methodology / evidence-vs-inference. One scenario is DIRECTIONAL — the 50–100 item
head-to-head is what actually decides Phase 12.
The comparison JSON is committable (it's small and it's the actual finding). To keep the raw runs,
download `evaluation/results/` from the Kaggle output panel before the session ends (Kaggle wipes
local storage on exit).

### Decision gate (Phase 12)
- **Deterministic anchor** (judge-independent) is the tie-breaker. If Moonlight ≈ Qwen there **and**
  the cross-judge doesn't overturn it → models are comparable; Qwen is **~3× faster** and Apache-2.0.
  Either keep Moonlight frozen or adopt Qwen — both are defensible; record the reason.
- If the cross-judge shows a **clear, consistent** gap (both directions agree) → that's the new
  signal; follow it.

---

## 2. LOCAL (this PC, no GPU) — live web + lab verification

### 2a. web tools (need internet; run in the LLM venv)
```bash
.venv/Scripts/python.exe scripts/try_web_fetch.py "https://example.com/"
```
✅ already verified working. `try_web_search.py` will print **BACKEND BLOCKED** from this IP — expected;
retry from a residential network or wire a keyed search API before trusting `web_search` live.

### 2b. operator loop against the DVWA lab (needs Docker Desktop up)
```bash
docker compose -f lab/docker-compose.yml up -d
.venv/Scripts/python.exe scripts/try_http_get.py "http://127.0.0.1:8080/login.php"
```
`http_get` / `nmap` / `ffuf` were live-tested against this lab earlier; re-run after any web-layer
change. Security tools stay gated by the **explicit operator target list**, never by IP class.

---

## 3. NOT pending (so nothing here surprises you later)
- No fine-tuning / SFT / DPO run is queued — by decision, not omission (see §0).
- `web_search` live-from-datacenter-IP is a **known** limitation, already surfaced in code.
- Integration TODO (deferred, optional): wire the memory store + assessment graph into the live
  `operator_loop` so a real session remembers and reasons over accumulated observations.
