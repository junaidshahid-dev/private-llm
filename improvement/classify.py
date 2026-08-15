"""classify.py — route a failure to the LAYER that should be fixed, so we fix the right thing.

    from improvement.classify import classify_failure
    result = classify_failure(record)     # record = a graded item / a session outcome

The project has already hit every one of these failure classes, and the expensive lesson was the
SFT rounds: changing the MODEL WEIGHTS when the real problem lived in RAG or in evaluation. This
classifier reads the signals we already record (scores, divergence, retrieval, tool results,
verification findings, the answer text) and names the layer to fix:

    model_reasoning | missing_knowledge | retrieval | tool_selection | tool_execution |
    verification | prompt_policy | evaluator

It is RULE-BASED on observable signals, not an LLM guess — a triage tool has to be reliable. And it
is conservative about blaming the model: an unreliable grader (evaluator) or an errored tool
(tool_execution) is diagnosed FIRST, because "the model reasoned wrong" is only credible once you
have confirmed the failure is real and not an artifact of a lower layer. It never recommends blanket
fine-tuning — at most a TARGETED example or a stronger base model.
"""
from __future__ import annotations

# (priority, layer): lower priority number wins the PRIMARY slot. Evaluator/tool-execution are
# diagnosed before the model, because they can manufacture a fake "model failure".
_REFUSAL = ("i cannot", "i can't", "i am unable", "i'm unable", "cannot help", "can't help",
            "unable to help", "as an ai", "i do not have access", "i don't have access")


def classify_failure(record: dict, available_tools=None) -> dict:
    score = record.get("score")
    det = record.get("det_score")
    div = record.get("divergence")
    judge_detail = (record.get("judge_detail") or "").lower()
    findings = " ".join(record.get("verify_findings") or []).lower()
    answer = (record.get("output") or record.get("final") or "").lower()
    verdict = record.get("verify_verdict", "PASS")
    tool_results = record.get("tool_results") or []
    grounded = record.get("grounded")

    signals: list[tuple[int, str, str]] = []

    # ---- layers that can FAKE a model failure — diagnose first ----------------
    if score is None or "unparsed" in judge_detail or "did not parse" in judge_detail:
        signals.append((0, "evaluator",
                        "the judge output did not parse — fix/replace the grader before trusting "
                        "this result; there is no reliable score yet"))
    if div is not None and div >= 0.34:
        signals.append((0, "evaluator",
                        f"judge and deterministic graders diverge ({div}) — confirm the TRUE score "
                        "by human review before changing anything downstream"))
    if any(not (t.get("result") or {}).get("ok") for t in tool_results):
        errs = [t.get("tool") for t in tool_results if not (t.get("result") or {}).get("ok")]
        signals.append((1, "tool_execution",
                        f"a tool call errored ({', '.join(map(str, errs))}) — fix the tool, its "
                        "arguments, or the environment (path, target authorization, binary), NOT "
                        "the model"))
    if "fabricate" in findings or "phantom" in findings:
        signals.append((1, "prompt_policy",
                        "the model fabricated a result or claimed an action it did not run — "
                        "tighten the reasoning prompt's honesty rules, not the weights"))

    failed = score is not None and score < 0.5

    # ---- capability failures (only when the score is actually low) ------------
    if failed:
        if any(p in answer for p in _REFUSAL):
            signals.append((2, "prompt_policy",
                            "the model refused or hedged a legitimate task — fix the reasoning "
                            "prompt/policy (capability-first), not the weights"))
        if grounded and ("grounding" in findings or record.get("kept_after_gate", 0)):
            signals.append((2, "retrieval",
                            "the answer was grounded on retrieved content and still failed — the "
                            "retrieval layer surfaced unhelpful/tangential material; fix relevance "
                            "gating, not the model"))
        if available_tools is not None:
            proposed = record.get("proposed_tools") or []
            bad = [t for t in proposed if t not in available_tools]
            if bad:
                signals.append((2, "tool_selection",
                                f"proposed a tool not available ({', '.join(bad)}) — fix the tool "
                                "schema/selection, not the model"))
        if grounded is False and not tool_results:
            signals.append((3, "missing_knowledge",
                            "neither the model nor the knowledge base had what was needed — add "
                            "authoritative, provenance-tracked knowledge for this topic"))
        # default capability bucket: the model's own reasoning
        if not any(s[1] in ("retrieval", "missing_knowledge", "prompt_policy", "tool_selection",
                            "tool_execution") for s in signals):
            signals.append((4, "model_reasoning",
                            "no tool/retrieval/knowledge/prompt signal — the model's own reasoning "
                            "failed; use a TARGETED counter-example or a stronger base model, NOT "
                            "blanket fine-tuning (small-scale SFT already degraded this model)"))
        # verification is always a SECONDARY note, never the primary fix for a capability failure
        if verdict == "PASS":
            signals.append((6, "verification",
                            "verification passed a low-scoring answer — consider a new deterministic "
                            "check so this failure class is caught automatically next time"))

    if not signals:
        return {"failure": False, "primary": None, "layers": [],
                "recommendation": "no failure detected (score is acceptable and no signal fired). "
                "A clean pass is not proof of correctness — spot-check periodically."}

    signals.sort(key=lambda s: s[0])
    primary = signals[0][1]
    layers = [{"layer": lyr, "reason": reason} for _, lyr, reason in signals]
    return {"failure": True, "primary": primary, "layers": layers,
            "recommendation": signals[0][2]}


def render(result: dict) -> str:
    if not result["failure"]:
        return "FAILURE CLASSIFIER: " + result["recommendation"]
    lines = [f"FAILURE CLASSIFIER -> fix the [{result['primary']}] layer first"]
    for i, lyr in enumerate(result["layers"]):
        tag = "PRIMARY" if i == 0 else "also"
        lines.append(f"  [{tag}] {lyr['layer']}: {lyr['reason']}")
    return "\n".join(lines)
