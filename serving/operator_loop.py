"""operator_loop.py — run the Phase-8 loop LIVE, with the real Moonlight.

    python serving/operator_loop.py "Inspect the repo and tell me the pinned model revision."
    python serving/operator_loop.py                 # then type the instruction when prompted

This is session.run_session wired to the actual model instead of a scripted stub: the diagrammed
loop, executed for real —

    your instruction
      -> Moonlight PLANS (proposes tools as inert JSON; nothing runs)
      -> each proposal printed; YOU type y/n            <- the human gate, the only thing that
      -> MCP executes ONLY what you approved               can cause execution
      -> Moonlight INTERPRETS the real results
      -> verification classifies the final answer
      -> final response + verdict

The approver is interactive and FAILS SAFE: anything other than an explicit 'y'/'yes' is a deny.
It defaults to a read-only tool config (configs/tools.yaml), so a fresh run cannot touch anything
destructive even if you approve — you raise the tool's permission in the YAML deliberately, not by
accident. Needs the GPU (the model); the loop logic itself is already CPU-tested in
mcp_layer/session_test.py.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace",
                              line_buffering=True)
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")


def interactive_approver(proposal: dict) -> bool:
    """Print the proposal and read the operator's decision. Fail safe: only an explicit yes runs."""
    print("\n" + "-" * 74)
    print("PROPOSED TOOL (nothing has run):")
    print(f"  tool : {proposal.get('tool')}")
    print(f"  args : {json.dumps(proposal.get('arguments', {}))}")
    print(f"  why  : {proposal.get('why', '')}")
    print(f"  kind : {proposal.get('kind', 'standard')}")
    try:
        reply = input("  approve and execute? [y/N] ").strip().lower()
    except EOFError:
        reply = ""
    ok = reply in ("y", "yes")
    print(f"  -> {'APPROVED' if ok else 'DENIED'}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("instruction", nargs="*", help="what to ask; prompted if omitted")
    ap.add_argument("--policy", default=None, help="behaviour mode override")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--auto-deny", action="store_true",
                    help="deny every proposal automatically (dry run: see the plan, run nothing)")
    ap.add_argument("--yes", action="store_true",
                    help="auto-approve READ-ONLY (standard) proposals — for notebooks where the "
                    "y/N prompt has no keyboard. Security-kind tools are STILL denied and need the "
                    "interactive gate; tools.yaml permissions still apply.")
    args = ap.parse_args()

    instruction = " ".join(args.instruction).strip()
    if not instruction:
        try:
            instruction = input("instruction> ").strip()
        except EOFError:
            instruction = ""
    if not instruction:
        print("no instruction given.")
        return 1

    from mcp_layer.session import run_session, render_session
    from serving.policy import system_prompt

    import torch
    if not torch.cuda.is_available():
        print("NO GPU — this drives the real model, so it needs one. The loop logic is CPU-tested "
              "in mcp_layer/session_test.py; run this where Moonlight is loaded.")
        return 2

    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from training.patches import apply_all
    if int(transformers.__version__.split(".")[0]) >= 5:
        sys.exit("transformers 5.x cannot quantise this model; install 4.57.6.")
    apply_all(verbose=False)

    lock = json.load(open(os.path.join(HERE, "MODEL_SPEC.lock.json"), encoding="utf-8"))
    q = lock["quantization"]
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_use_double_quant=q["double_quant"],
                             bnb_4bit_compute_dtype=torch.float16)
    print("loading base Moonlight (4-bit)...")
    model = AutoModelForCausalLM.from_pretrained(
        lock["model"], revision=lock["revision"], quantization_config=bnb, device_map={"": 0})
    model.eval()
    tok = AutoTokenizer.from_pretrained(lock["model"], revision=lock["revision"],
                                        trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    def generate(messages) -> str:
        ids = tok.apply_chat_template(messages, add_generation_prompt=True,
                                      return_tensors="pt").to(0)
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=args.max_new_tokens, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        return tok.decode(out[0][ids.shape[-1]:], skip_special_tokens=True).strip()

    def auto_yes(proposal: dict) -> bool:
        # notebook-safe: approve read-only standard tools, never a security tool unattended
        ok = proposal.get("kind", "standard") == "standard"
        print(f"\n[--yes] {proposal.get('tool')} ({proposal.get('kind','standard')}) -> "
              f"{'APPROVED (read-only)' if ok else 'DENIED (security tool needs interactive gate)'}")
        return ok

    approver = (lambda _p: False) if args.auto_deny else (auto_yes if args.yes else
                                                          interactive_approver)
    record = run_session(instruction, generate, approver,
                         policy_prompt=system_prompt(args.policy))
    print("\n" + "=" * 74)
    print(render_session(record))
    return 0


if __name__ == "__main__":
    sys.exit(main())
