"""assessment.py — run a full AUTONOMOUS authorized assessment end to end.

The operator starts an authorized session (target + objective + capability profile) ONCE; the agent
then runs the plan -> (autonomous approve via the session policy) -> execute -> interpret -> verify
loop with NO per-tool prompts, feeds every real result through the discovery pipeline (trust
boundary + assessment graph + evidence-graded findings + telemetry), and produces a professional
report. The kill switch, the authorized-target scope, and the capability profile remain hard gates.

`generate` is injectable, so the whole orchestration is CPU-testable with a stub model; the __main__
path loads the real model on the GPU. This is the wiring; every component it calls is already tested.
"""
from __future__ import annotations

from research.findings import Evidence, Hypothesis


def _hyp_from_finding(f: dict) -> Hypothesis:
    """Reconstruct a research Hypothesis from a tool's structured finding (source_scan/config_scan).
    Evidence kind='code' -> the status derives to POSSIBLE/LIKELY, never CONFIRMED without a test."""
    comp = f.get("component") or f.get("affected_component") or "source"
    return Hypothesis(title=f.get("title", "candidate finding"), vuln_class=f.get("vuln_class", ""),
                      affected_component=comp, severity=f.get("severity", "INFO"),
                      why_it_matters="static-analysis candidate",
                      next_test="validate that the sink is reachable with attacker-controlled input",
                      alternative_explanation="the input may be sanitised or the path unreachable",
                      evidence=[Evidence(comp, "code", f.get("title", ""), 0.6)])


def run_assessment(session, generate, *, config=None, executor=None, verify_fn=None,
                   max_rounds=8) -> dict:
    """Drive one autonomous authorized assessment. Returns {record, findings, report, telemetry}."""
    from mcp_layer import permissions as perm
    from mcp_layer import session as sess_mod
    from mcp_layer import session_policy
    from mcp_layer.telemetry import Telemetry
    from research.pipeline import DiscoveryPipeline

    cfg = config or perm.load_config()
    tel = Telemetry(getattr(session, "id", "session"))
    pipe = DiscoveryPipeline(session, telemetry=tel)
    approver = session_policy.approver_for(session, cfg)          # autonomous: no per-call prompt

    record = sess_mod.run_session(session.objective, generate, approver=approver, config=cfg,
                                  executor=executor, verify_fn=verify_fn, max_rounds=max_rounds,
                                  telemetry=tel)

    # feed the REAL tool results into the discovery pipeline (state + timeline + findings)
    for r in record.get("results", []):
        res = r.get("result", {}) or {}
        pipe.observe(r.get("tool", ""), "", {}, res)
        for f in (res.get("findings") or []):
            pipe.add_finding(_hyp_from_finding(f))

    report = pipe.report()
    return {"record": record, "findings": pipe.findings, "confirmed": len(pipe.confirmed()),
            "report": report, "telemetry": tel.chain(), "escalated": record.get("escalated")}


def main() -> int:
    import argparse
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    except (AttributeError, ValueError):
        pass
    from mcp_layer import session_policy

    ap = argparse.ArgumentParser(description="Run an autonomous authorized assessment.")
    ap.add_argument("objective")
    ap.add_argument("--target", action="append", default=[], help="authorized target (repeatable)")
    ap.add_argument("--profile", default="recon", choices=list(session_policy.PROFILES))
    ap.add_argument("--time-limit", type=int, default=3600)
    ap.add_argument("--max-rounds", type=int, default=8)
    ap.add_argument("--model")
    args = ap.parse_args()

    # running THIS CLI is the operator's explicit authorization (operator_ack=True).
    started = session_policy.start_session(args.objective, args.target, args.profile,
                                           args.time_limit, operator_ack=True)
    if not started["ok"]:
        print("cannot start session:", started["error"])
        return 2
    session = started["session"]
    print(session.render())

    import torch
    if not torch.cuda.is_available():
        print("\nNO GPU — this live runner needs the model. Run on a GPU host. "
              "(The orchestration is unit-tested on CPU via serving/assessment_test.py.)")
        return 2
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from training.patches import apply_all
    from serving.model_spec import load_lock
    from serving.policy import system_prompt
    apply_all(verbose=False)
    lock = load_lock(args.model)
    q = lock["quantization"]
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_use_double_quant=q["double_quant"],
                             bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(lock["model"], revision=lock["revision"],
                                                 quantization_config=bnb, device_map={"": 0})
    model.eval()
    tok = AutoTokenizer.from_pretrained(lock["model"], revision=lock["revision"],
                                        trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    def generate(messages):
        ids = tok.apply_chat_template(messages, add_generation_prompt=True,
                                      return_tensors="pt").to(0)
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=768, do_sample=False, pad_token_id=tok.pad_token_id)
        return tok.decode(out[0][ids.shape[-1]:], skip_special_tokens=True).strip()

    result = run_assessment(session, generate, max_rounds=args.max_rounds)
    print("\n" + result["report"])
    print(f"\n[{result['confirmed']} confirmed / {len(result['findings'])} findings; "
          f"{len(result['telemetry'])} telemetry events]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
