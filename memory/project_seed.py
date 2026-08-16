"""project_seed.py — seed the memory store with the REAL private-llm project history (Phase 11.9).

So the assistant can answer "why did we stop SFT?" or "what's the current baseline?" from actual
recorded decisions/experiments, not by guessing. These are durable project facts (semantic),
recorded experiments (episodic), and the how-we-work rule (procedural), in project "private-llm".

    from memory.store import MemoryStore
    from memory.project_seed import seed
    seed(MemoryStore(project="private-llm"))
"""
from __future__ import annotations

# (text, type, importance)
FACTS = [
    ("The private-llm architecture is model-agnostic: base model + RAG + MCP tools + explicit "
     "operator approval + verification + memory + web layer + a security lab. Swapping the model "
     "touches only MODEL_SPEC.lock.json (via serving/model_spec.load_lock), nothing else.",
     "semantic", 0.95),
    ("We stopped small-scale SFT because it MEASURABLY degraded Moonlight twice (objective "
     "0.594->0.478->0.377, style collapse, math -0.533). The fix is capability at inference "
     "(RAG + the behaviour policy), not retraining. Do not retrain by reflex.",
     "episodic", 0.95),
    ("Moonlight-16B-A3B-Instruct is the frozen baseline (git tag moonlight-baseline). We A/B a "
     "stronger base model instead of prompt-patching, because prompt fixes hit the 16B ceiling.",
     "semantic", 0.9),
    ("Security v3 head-to-head: the self-judge rated Moonlight 1.000 vs Qwen 0.795, but the "
     "unbiased deterministic grader said 0.675 vs 0.645 (tied) — the self-judge has home-field "
     "bias. Models are comparable; Qwen2.5-Coder-14B is ~3x faster. Verdict pending a cross-judge.",
     "episodic", 0.85),
    ("RAG HURT out-of-domain (grounding on tangential docs) until a relevance gate was added. RAG's "
     "real value is niche/current/project-specific facts, not general knowledge the model already "
     "has. base Moonlight is strong on standard security topics on its own.",
     "episodic", 0.85),
    ("The self-judge is unreliable for ranking models (home-field bias); use deterministic anchors "
     "plus an INDEPENDENT/cross judge (evaluation/judge_pass.py). This is the blocker for the "
     "Phase-12 model decision.",
     "semantic", 0.85),
    ("Security tools are authorized by an explicit operator target list (NOT by IP class). Web "
     "tools are SSRF-defended (public-only by default) and treat page content as UNTRUSTED data, "
     "never instructions. Secrets are never stored in memory.",
     "semantic", 0.85),
    ("Model history: base = Moonlight-16B-A3B-Instruct (MoE, DeepSeek-V3 arch, transformers pinned "
     "4.57.6, T4 4-bit ~8.5GB). First swap candidate = Qwen2.5-Coder-14B-Instruct (dense, ~9GB, "
     "faster, Apache-2.0).",
     "semantic", 0.8),
    ("Roadmap: phases 1-11 done (base, policy, RAG, MCP, approval, security KB, benchmarks, "
     "verification, operator loop, web layer, memory) plus the assessment state graph. Phase 12 = "
     "stronger base model. Phase 13 = own model.",
     "semantic", 0.8),
    ("When evaluating models: (1) benchmark on HELD-OUT items, (2) inspect PER-ITEM failures, not "
     "just the aggregate, (3) distrust a self-judge, (4) fix the RIGHT layer (classify the failure: "
     "model/RAG/tool/prompt/eval) instead of retraining.",
     "procedural", 0.9),
    ("The operator prefers verified, honest answers over confident guesses, and wants mistakes "
     "DETECTED and surfaced, not hidden. 'Done' means implemented + permissioned + tested + "
     "measured + a known failure mode — not 'can do anything'.",
     "procedural", 0.95),
]


def seed(store, project="private-llm") -> dict:
    """Add each project fact if not already present (dedup). Returns a count."""
    from memory.extract import _is_duplicate
    added = 0
    for text, mtype, imp in FACTS:
        if not _is_duplicate(text, store, project):
            r = store.add(text, mtype=mtype, importance=imp, project=project, source="project_seed")
            added += 1 if r.get("ok") else 0
    return {"ok": True, "added": added, "total": len(FACTS), "project": project}
