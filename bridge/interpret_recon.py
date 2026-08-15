"""interpret_recon.py — Moonlight interprets the REAL tool output captured locally (run on Kaggle).

    python bridge/interpret_recon.py                      # reads bridge/recon_capture.json
    python bridge/interpret_recon.py --capture other.json

The other half of the bridge. capture_recon.py produced real nmap/ffuf results on the machine with
Docker; this loads the model on the machine with the GPU and has it do the INTERPRET step of the
loop over that real data — the analysis a human did by hand, now done by Moonlight. Verification
runs on the model's analysis with the real tool_results, so if it invents a finding not present in
the output, the fabricated-tool-output check flags it.

This closes the model-INTERPRET half of the co-location gap without needing Docker on the GPU box.
It does NOT run tools (there are none here) — it reasons over captured evidence. Needs the GPU.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

DEFAULT_CAPTURE = os.path.join(HERE, "bridge", "recon_capture.json")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", default=DEFAULT_CAPTURE)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    args = ap.parse_args()

    if not os.path.exists(args.capture):
        sys.exit(f"no capture at {args.capture} — run bridge/capture_recon.py locally, push, pull.")
    cap = json.load(open(args.capture, encoding="utf-8"))
    results = cap.get("results", [])
    task = cap.get("task", "Interpret these security tool results.")
    print("=" * 74)
    print(f"BRIDGE — Moonlight interprets REAL recon of {cap.get('target', '?')}")
    print(f"{len(results)} tool result(s) captured locally; the model reasons over them here.")
    print("=" * 74)
    for r in results:
        res = r.get("result", {})
        print(f"  {r.get('tool')}: {'ok' if res.get('ok') else 'error'} "
              f"({len(str(res.get('output', '')))} chars)")

    from mcp_layer import controller
    from serving.policy import system_prompt
    from verification.verify import verify

    import torch
    if not torch.cuda.is_available():
        print("\nNO GPU — this needs the model. Run on Kaggle. (Capture is machine-independent.)")
        return 2

    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from training.patches import apply_all
    if int(transformers.__version__.split(".")[0]) >= 5:
        sys.exit("transformers 5.x cannot quantise this model; install 4.57.6.")
    apply_all(verbose=False)
    from serving.model_spec import load_lock
    lock = load_lock()
    q = lock["quantization"]
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_use_double_quant=q["double_quant"],
                             bnb_4bit_compute_dtype=torch.float16)
    print("\nloading base Moonlight (4-bit)...")
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

    # the INTERPRET step of the loop, over REAL captured results
    analysis = controller.interpret(task, results, generate, policy_prompt=system_prompt())
    report = verify(analysis, hits=None, tools_ran=[r.get("tool") for r in results],
                    tool_results=results)

    print("\n" + "=" * 74)
    print("MOONLIGHT'S ANALYSIS (over the real tool output):")
    print("=" * 74)
    print(analysis)
    print("\n" + "-" * 74)
    print(f"Verification: {report.verdict}")
    for f in report.findings:
        print(f"  - {f}")
    print(f"  next: {report._NEXT[report.verdict]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
